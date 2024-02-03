import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import boto3
import pickle
import gzip
import io
from st_aggrid import AgGrid, GridOptionsBuilder
from supabase import create_client, Client

# Initialize Supabase client using Streamlit secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase = create_client(supabase_url, supabase_key)

def load_model_from_s3(bucket_name, model_key):
    # Function to load a model from S3
    s3 = boto3.client('s3', aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
                      aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"])
    response = s3.get_object(Bucket=bucket_name, Key=model_key)
    model_str = response['Body'].read()
    model = pickle.loads(gzip.decompress(model_str))
    return model

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
    # Data preprocessing logic
    df = df.copy()
    # Example preprocessing steps
    for col in ['payment_type', 'source', 'device_os', 'employment_status', 'housing_status']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            df[col] = LabelEncoder().fit_transform(df[col])
    return df

def run_inference(transactions_data, rf_model, lof_model):

    # Store 'ref_id' in a separate Series to preserve the order
    ref_ids = transactions_data['ref_id'].copy()
    
    # Preprocess the data, excluding 'ref_id' for the prediction
    preprocessed_data = preprocess_data(transactions_data.drop(columns=['ref_id']))
    
    # Retrieve user-defined settings
    fraud_threshold = st.session_state.get('fraud_threshold', 0.5)  # Default to 0.5 if not set
    
    # Predict potential fraud cases with probabilities
    rf_probabilities = rf_model.predict_proba(preprocessed_data)[:, 1]
    rf_predictions = [1 if prob > fraud_threshold else 0 for prob in rf_probabilities]

    # Filter out transactions flagged as potential fraud and non-fraud
    potential_fraud_indices = [i for i, pred in enumerate(rf_predictions) if pred == 1]
    potential_nonfraud_indices = [i for i, pred in enumerate(rf_predictions) if pred == 0]
    X_potential_nonfraud = preprocessed_data.iloc[potential_nonfraud_indices]

    # Apply LOF model on potential non-fraud cases to further screen for anomalies
    # Note: Ensure your LOF model was initialized with novelty=True for this use case
    lof_predictions = lof_model.predict(X_potential_nonfraud)
    
    # Identify indices of anomalies detected by LOF within the non-fraud cases
    lof_anomaly_indices = [potential_nonfraud_indices[i] for i, pred in enumerate(lof_predictions) if pred == -1]

    # Combine indices for review or further action
    combined_review_indices = set(potential_fraud_indices + lof_anomaly_indices)

    return combined_review_indices


def create_combined_flags_table(combined_flags_indices, transactions_data, selected_features):
    # Creating a combined flags table
    combined_flags_data = transactions_data.loc[combined_flags_indices]
    combined_flags_data['flag'] = 'Potential Fraud/Anomaly'
    return combined_flags_data

def display_transactions_data(transactions_data):
    # Using AgGrid to display transactions data
    gb = GridOptionsBuilder.from_dataframe(transactions_data)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    grid_options = gb.build()
    AgGrid(transactions_data, gridOptions=grid_options, enable_enterprise_modules=True)

def display_combined_data(unified_flags, anomaly_detection_records):
    # Displaying unified flags and anomaly detection records if available
    if unified_flags or anomaly_detection_records:
        st.write("Combined Flags and Anomaly Detection Table:")
        combined_table = pd.concat([pd.DataFrame(unified_flags), pd.DataFrame(anomaly_detection_records)], ignore_index=True)
        st.dataframe(combined_table)

# Initialize offline_review_transactions variable
offline_review_transactions = set()

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

    # Define combined_flags_table and initialize it as None
    combined_flags_table = None

    # Button to run preprocessing and inference
    if not transactions_data.empty:
        if st.button('Run Preprocessing and Inference'):
            with st.spinner('Running preprocessing and inference...'):
                # Assuming you allow users to select features in settings
                selected_features = st.session_state.get('selected_features', transactions_data.columns.tolist())
                preprocessed_data = preprocess_data(transactions_data[selected_features])
                
                # Run inference with the preprocessed data and loaded models
                #run_inference(transactions_data, rf_model, lof_model, selected_features)  # Pass selected_features here
                run_inference(transactions_data, rf_model, lof_model)  # Adjusted call with 3 arguments

                
        # Display transaction data in an interactive grid
        gb = GridOptionsBuilder.from_dataframe(transactions_data)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
        grid_options = gb.build()
        AgGrid(transactions_data, gridOptions=grid_options, enable_enterprise_modules=True)
        
        # Debug: Print offline_review_indices
        offline_review_indices = st.session_state.get('offline_review_transactions', [])
        #st.write("Debug: Offline Review Indices:", offline_review_indices)

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

        # Display the Unified Flags table and the Anomaly Detection table
        unified_flags = st.session_state.get('unified_flags', [])
        anomaly_detection_records = st.session_state.get('anomaly_detection_records', [])
        
        if unified_flags or anomaly_detection_records:
            st.write("Combined Flags and Anomaly Detection Table:")
            combined_table = pd.concat([pd.DataFrame(unified_flags), pd.DataFrame(anomaly_detection_records)], ignore_index=True)
            st.write(combined_table)
            
        # Store the combined flags table in session_state
        st.session_state['combined_flags_table'] = combined_flags_table

        # Store the combined flags table in session_state after ensuring it's a DataFrame
        if isinstance(combined_flags_table, pd.DataFrame):
            st.session_state['combined_flags_table'] = combined_flags_table
            # Immediately verify
            if isinstance(st.session_state['combined_flags_table'], pd.DataFrame):
                st.write("DataFrame stored correctly in session state.")
            else:
                st.error("Failed to store DataFrame correctly in session state.")

if __name__ == '__main__':
    transactions_page()


