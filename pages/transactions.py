import streamlit as st
import pandas as pd
import pickle
import gzip
import io
import datetime
import boto3
from sklearn.preprocessing import LabelEncoder
from st_aggrid import AgGrid, GridOptionsBuilder
from supabase import create_client, Client

# Initialize Supabase client using Streamlit secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize offline_review_transactions variable
offline_review_transactions = set()

# Define an empty list to store audit logs
audit_logs = []

# Define unified_flags and anomaly_detection_records
unified_flags = []
anomaly_detection_records = []

# Function to log an entry in the audit logs
def log_audit_entry(transaction_id, reviewer_id, decision):
    timestamp = datetime.datetime.now()
    audit_entry = {
        'Timestamp': timestamp,
        'Transaction ID': transaction_id,
        'Reviewer ID': reviewer_id,
        'Decision': decision,
    }
    audit_logs.append(audit_entry)

def load_model_from_s3(bucket_name, model_key):
    aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
    aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
    model_str = response['Body'].read()
    with gzip.GzipFile(fileobj=io.BytesIO(model_str)) as file:
        return pickle.load(file)

def fetch_transactions():
    try:
        response = supabase.table('transactions').select('*').execute()
        if hasattr(response, 'error') and response.error:
            st.error(f'Failed to retrieve data. Error: {str(response.error)}')
            return pd.DataFrame()
        elif hasattr(response, 'data'):
            return pd.DataFrame(response.data)
        else:
            st.error('Unexpected response format.')
            return pd.DataFrame()
    except Exception as e:
        st.error(f'An error occurred: {e}')
        return pd.DataFrame()

def preprocess_data(df):
    df = df.drop(columns=['ref_id'], errors='ignore')
    categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    for col in categorical_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = df[col].fillna('Unknown')
            df[col] = encoder.fit_transform(df[col])
    for col in df.columns:
        if df[col].dtype != 'O':
            df[col] = df[col].fillna(df[col].median())
    return df

def run_inference(transactions_data, rf_model, lof_model, selected_features):
    # Preprocess the data
    preprocessed_data = preprocess_data(transactions_data[selected_features])
    
    # Retrieve user-defined settings
    fraud_threshold = st.session_state.get('fraud_threshold', 0.5)  # Default to 0.5 if not set

    # Predict potential fraud cases with probabilities
    selected_features = [feat for feat in selected_features if feat != 'ref_id']
    rf_probabilities = rf_model.predict_proba(preprocessed_data)[:, 1]
    rf_predictions = [1 if prob > 0.5 else 0 for prob in rf_probabilities]

    # Filter out transactions flagged as potential fraud and non-fraud
    potential_fraud_indices = [i for i, pred in enumerate(rf_predictions) if pred == 1]
    potential_nonfraud_indices = [i for i, pred in enumerate(rf_predictions) if pred == 0]
    X_potential_nonfraud = preprocessed_data.iloc[potential_nonfraud_indices]

    # Apply LOF model on potential non-fraud cases
    lof_predictions = lof_model.fit_predict(X_potential_nonfraud)
    lof_anomaly_indices = [potential_nonfraud_indices[i] for i, pred in enumerate(lof_predictions) if pred == -1]

    # Display LOF anomaly indices in Streamlit app
    st.write("LOF Anomaly Indices:", lof_anomaly_indices)

    # Store results in session_state for later access in expert_human_judgment_page.py
    st.session_state['potential_fraud_indices'] = potential_fraud_indices
    st.session_state['lof_anomaly_indices'] = lof_anomaly_indices

    # Combine LOF anomalies and RF frauds for human review
    offline_review_transactions = set(potential_fraud_indices + lof_anomaly_indices)
    st.session_state['offline_review_transactions'] = list(offline_review_transactions)
    # Display the value of offline_review_transactions
    st.write("Offline Review Transactions:", offline_review_transactions)

    # Prepare data for Unified Flags and Anomaly Detection Tables
    unified_flags, anomaly_detection_records = [], []
    for index in range(len(transactions_data)):
        transaction_record = transactions_data.iloc[index].to_dict()
        
        # Check the structure of transaction_record
        if 'ref_id' in transaction_record:
            ref_id = transaction_record['ref_id']
        else:
            # Handle the case where 'ref_id' is not present in the dictionary
            ref_id = None  # Or use an appropriate default value
        
        if index in potential_fraud_indices:
            # Add to unified flags if RF model predicts fraud
            unified_flags.append({
                'flag_id': ref_id,
                'model_version': 'RF_v1',
                'prob_score': rf_probabilities[index],
                'flag_type': 'possible fraud',
                **transaction_record  # Include original transaction data
            })
        
        if index in lof_anomaly_indices:
            # Correctly identify the LOF model index
            lof_model_index = X_potential_nonfraud.index.get_loc(index)
            anomaly_score = -lof_model.negative_outlier_factor_[lof_model_index]
            anomaly_detection_record = {
                'anomaly_id': ref_id,
                'model_version': 'LOF_v1',
                'anomaly_score': anomaly_score,
                'flag_type': 'possible fraud',  # Flag type for anomaly is also 'fraud'
                'is_anomaly': True,
                **transaction_record  # Include original transaction data
            }
            anomaly_detection_records.append(anomaly_detection_record)

    # Set the 'offline_review_transactions' variable in the session_state
    st.session_state['offline_review_transactions'] = offline_review_transactions  # Pass it to session_state

    # After running models, store results in session_state
    st.session_state['transactions_data'] = transactions_data
    st.session_state['preprocessed_data'] = preprocessed_data
    st.session_state['rf_predictions'] = rf_predictions
    st.session_state['lof_anomaly_indices'] = lof_anomaly_indices

    # Assuming 'unified_flags' and 'anomaly_detection_records' are your final outputs
    st.session_state['unified_flags'] = unified_flags
    st.session_state['anomaly_detection_records'] = anomaly_detection_records
    st.write("Potential Fraud Indices:", potential_fraud_indices)
    st.write("LOF Anomaly Indices:", lof_anomaly_indices)

    st.success("Inference complete. Go to the offline review page to view transactions for review.")


def create_offline_review_table(offline_review_indices, transactions_data, rf_model, lof_model, selected_features):
    
    # Exclude 'ref_id' from selected features
    selected_features = [feat for feat in selected_features if feat != 'ref_id']
    
    table_data = []

    # Retrieve RF probabilities once
    rf_probabilities = rf_model.predict_proba(transactions_data[selected_features])[:, 1]

    for index in offline_review_indices:
        transaction_record = transactions_data.iloc[index].to_dict()

        # Check if 'ref_id' key exists in the dictionary
        if 'ref_id' in transaction_record:
            ref_id = transaction_record['ref_id']
        else:
            ref_id = None  # Handle the case where 'ref_id' is not present

        if index in potential_fraud_indices:
            # Add to unified flags if RF model predicts fraud
            unified_flags.append({
                'flag_id': ref_id,
                'model_version': 'RF_v1',
                'prob_score': rf_probabilities[index],
                'flag_type': 'possible fraud',
                **transaction_record  # Include original transaction data
            })

        if index in lof_anomaly_indices:
            # Correctly identify the LOF model index
            lof_model_index = X_potential_nonfraud.index.get_loc(index)
            anomaly_score = -lof_model.negative_outlier_factor_[lof_model_index]
            anomaly_detection_record = {
                'anomaly_id': ref_id,
                'model_version': 'LOF_v1',
                'anomaly_score': anomaly_score,
                'flag_type': 'possible fraud',  # Flag type for anomaly is also 'fraud'
                'is_anomaly': True,
                **transaction_record  # Include original transaction data
            }
            anomaly_detection_records.append(anomaly_detection_record)

    # Set the 'offline_review_transactions' variable in the session_state
    st.session_state['offline_review_transactions'] = offline_review_indices  # Pass it to session_state

    # Assuming 'unified_flags' and 'anomaly_detection_records' are your final outputs
    st.session_state['unified_flags'] = unified_flags
    st.session_state['anomaly_detection_records'] = anomaly_detection_records

    # Return the table data
    return table_data

    
def transactions_page():
    st.set_page_config(layout="wide")
    st.title('Transactions')

    # Load models and fetch transactions
    bucket_name = 'frauddetectpred'
    rf_model_key = 'random_forest_model.pkl.gz'
    lof_model_key = 'lof_nonfraud.pkl.gz'
    rf_model = load_model_from_s3(bucket_name, rf_model_key)
    lof_model = load_model_from_s3(bucket_name, lof_model_key)

    transactions_data = fetch_transactions()

    # Initialize offline_review_transactions variable here
    offline_review_transactions = set()

    # Button to run preprocessing and inference
    if not transactions_data.empty:
        if st.button('Run Preprocessing and Inference'):
            with st.spinner('Running preprocessing and inference...'):
                # Assuming you allow users to select features in settings
                selected_features = st.session_state.get('selected_features', transactions_data.columns.tolist())
                preprocessed_data = preprocess_data(transactions_data[selected_features])
                
                # Run inference with the preprocessed data and loaded models
                run_inference(transactions_data, rf_model, lof_model, selected_features)  # Pass selected_features here
                
        # Display transaction data in an interactive grid
        gb = GridOptionsBuilder.from_dataframe(transactions_data)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
        grid_options = gb.build()
        AgGrid(transactions_data, gridOptions=grid_options, enable_enterprise_modules=True)
        
        # Display LOF anomaly indices separately
        lof_anomaly_indices = st.session_state.get('lof_anomaly_indices', [])
        if lof_anomaly_indices:
            st.write("LOF Anomaly Indices:", lof_anomaly_indices)

        # Combine LOF anomalies and RF frauds for human review
        potential_fraud_indices = st.session_state.get('potential_fraud_indices', [])
        lof_anomaly_indices = st.session_state.get('lof_anomaly_indices', [])
        
        # Convert potential_fraud_indices and lof_anomaly_indices to sets
        potential_fraud_set = set(potential_fraud_indices)
        lof_anomaly_set = set(lof_anomaly_indices)
        
        # Find the intersection of sets to get combined_flags_set
        combined_flags_set = potential_fraud_set.intersection(lof_anomaly_set)
        
        # Convert combined_flags_set back to a list for display
        combined_flags_indices = list(combined_flags_set)

        if combined_flags_indices:
            st.write("Combined Flags (Possible Fraud):", combined_flags_indices)
            
            # Create and display the combined flags table with modified columns
            combined_flags_table = create_combined_flags_table(combined_flags_indices, transactions_data, selected_features)
            st.write("Combined Flags Table:")
            st.write(combined_flags_table.rename(columns={'model_version': 'model_type', 'prob_score': 'score'}))

        # Display the offline review transactions table
        offline_review_indices = st.session_state.get('offline_review_transactions', [])
        if offline_review_indices:
            st.write("Offline Review Transactions:")
            offline_review_table = create_offline_review_table(offline_review_indices, transactions_data, rf_model, lof_model, selected_features)
            st.write(offline_review_table)

    else:
        st.error("No transactions data available.")

if __name__ == '__main__':
    transactions_page()
