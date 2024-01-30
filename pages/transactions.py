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

def run_inference(transactions_data, rf_model, lof_model):
    # Preprocess the data
    preprocessed_data = preprocess_data(transactions_data)

    # Predict potential fraud cases with probabilities
    rf_probabilities = rf_model.predict_proba(preprocessed_data)[:, 1]
    rf_predictions = [1 if prob > 0.5 else 0 for prob in rf_probabilities]

    # Filter out transactions flagged as potential fraud and non-fraud
    potential_fraud_indices = [i for i, pred in enumerate(rf_predictions) if pred == 1]
    potential_nonfraud_indices = [i for i, pred in enumerate(rf_predictions) if pred == 0]
    X_potential_nonfraud = preprocessed_data.iloc[potential_nonfraud_indices]

    # Apply LOF model on potential non-fraud cases
    lof_anomaly_indices = []
    if len(X_potential_nonfraud) > 20:
        lof_predictions = lof_model.fit_predict(X_potential_nonfraud)
        lof_anomaly_indices = [index for index, pred in zip(potential_nonfraud_indices, lof_predictions) if pred == -1]

    # Prepare data for Unified Flags Table
    unified_flags = []
    for index in potential_fraud_indices:
        unified_flags.append({
            'flag_id': transactions_data.iloc[index]['ref_id'],
            'ref_id': transactions_data.iloc[index]['ref_id'],
            'flagged_at': datetime.datetime.now(),
            'model_version': 'RF_v1',
            'flag_reason': rf_probabilities[index],
            'flag_type': 'fraud'
        })

    # Mapping original indices to LOF model indices
    lof_index_mapping = {original_index: lof_index for lof_index, original_index in enumerate(X_potential_nonfraud.index)}

    # Prepare data for Anomaly Detection Table
    anomaly_detection_records = []
    for original_index in lof_anomaly_indices:
        lof_model_index = lof_index_mapping.get(original_index, None)
        if lof_model_index is not None:
            anomaly_detection_records.append({
                'anomaly_id': transactions_data.iloc[original_index]['ref_id'],
                'ref_id': transactions_data.iloc[original_index]['ref_id'],
                'checked_at': datetime.datetime.now(),
                'anomaly_score': -lof_model.negative_outlier_factor_[lof_model_index],
                'threshold': 'LOF_v1'
            })

    # Save unified flags and anomaly detection records (implement saving logic as needed)

    st.session_state['rf_predictions'] = rf_predictions
    st.session_state['rf_probabilities'] = rf_probabilities
    st.session_state['potential_fraud_indices'] = potential_fraud_indices
    st.session_state['lof_anomaly_indices'] = lof_anomaly_indices
    st.success("Inference complete and results saved.")

    # Example of setting session state after inference
    #st.session_state['rf_model'] = rf_model
    #st.session_state['y_true'] = y_true  # This needs to come from your actual data
    #st.session_state['y_pred'] = rf_predictions
    # st.session_state['y_prob'] = rf_probabilities
    #st.session_state['model_metrics'] = {
        #'accuracy': accuracy_score(y_true, rf_predictions),
        #'precision': precision_score(y_true, rf_predictions),
        #'recall': recall_score(y_true, rf_predictions),
        #'f1_score': f1_score(y_true, rf_predictions)
    #}
    
    st.success("Inference complete. Go to the results page to view the results.")


def transactions_page():
    st.set_page_config(layout="wide")
    st.title('Transactions')

    bucket_name = 'frauddetectpred'
    rf_model_key = 'random_forest_model.pkl.gz'
    lof_model_key = 'lof_nonfraud.pkl.gz'

    rf_model = load_model_from_s3(bucket_name, rf_model_key)
    lof_model = load_model_from_s3(bucket_name, lof_model_key)

    transactions_data = fetch_transactions()
    if not transactions_data.empty:
        if st.button('Run Inference') and rf_model and lof_model:
            run_inference(transactions_data, rf_model, lof_model)

        gb = GridOptionsBuilder.from_dataframe(transactions_data)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
        grid_options = gb.build()
        AgGrid(transactions_data, gridOptions=grid_options, enable_enterprise_modules=True)
    else:
        st.write("No transactions data available.")

transactions_page()
