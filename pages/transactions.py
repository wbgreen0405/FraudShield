import streamlit as st
import pandas as pd
import pickle
import gzip
import io
import boto3
from sklearn.preprocessing import LabelEncoder
from st_aggrid import AgGrid, GridOptionsBuilder
from supabase import create_client, Client

# Initialize Supabase client using Streamlit secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase: Client = create_client(supabase_url, supabase_key)

def load_model_from_s3(bucket_name, model_key):
    # Access AWS credentials from Streamlit secrets
    aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
    aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]

    # Initialize S3 client with credentials
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # Get object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
    model_str = response['Body'].read()

    # Assuming the model is saved in a GZIP file
    with gzip.GzipFile(fileobj=io.BytesIO(model_str)) as file:
        return pickle.load(file)

def fetch_transactions():
    try:
        response = supabase.table('transactions').select('*').limit(100).execute()
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
    # Drop the 'ref_id' column as it's not used for predictions
    df = df.drop(columns=['ref_id'], errors='ignore')

    # Encode categorical columns
    categorical_cols = [
        'payment_type', 'employment_status', 'housing_status', 
        'source', 'device_os']
    for col in categorical_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = df[col].fillna('Unknown')  # Fill NaNs with 'Unknown'
            df[col] = encoder.fit_transform(df[col])

    # Handle any remaining missing values in numerical columns
    # Example: Fill NaNs with the median of the column
    for col in df.columns:
        if df[col].dtype != 'O':  # If column is not an object (text)
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
    X_potential_nonfraud = transactions_data.iloc[potential_nonfraud_indices]

    # Apply LOF model on potential non-fraud cases
    lof_anomaly_indices = []
    if len(X_potential_nonfraud) > 20:
        lof_predictions = lof_model.fit_predict(X_potential_nonfraud)
        lof_anomaly_indices = [index for index, pred in zip(potential_nonfraud_indices, lof_predictions) if pred == -1]

    # Combine LOF anomalies and RF frauds for human review
    offline_review_transactions = set(potential_fraud_indices + lof_anomaly_indices)

    # Prepare data for saving
    save_unified_flags(transactions_data.iloc[potential_fraud_indices], rf_predictions, rf_probabilities)
    save_anomaly_detection_records(transactions_data.iloc[lof_anomaly_indices], lof_anomaly_indices)

    st.session_state['rf_predictions'] = rf_predictions
    st.session_state['rf_probabilities'] = rf_probabilities
    st.session_state['potential_fraud_indices'] = potential_fraud_indices
    st.session_state['lof_anomaly_indices'] = lof_anomaly_indices

    st.success("Inference complete and results saved.")

    st.session_state['rf_predictions'] = rf_predictions
    st.session_state['rf_probabilities'] = rf_probabilities
    st.session_state['potential_fraud_indices'] = potential_fraud_indices
    st.session_state['lof_anomaly_indices'] = lof_anomaly_indices

    st.success("Inference complete and results saved.")


def transactions_page():
    st.set_page_config(layout="wide")
    st.title('Transactions')

    # Define AWS S3 bucket and model keys
    bucket_name = 'frauddetectpred'
    rf_model_key = 'random_forest_model.pkl.gz'
    lof_model_key = 'lof_nonfraud.pkl.gz'

    # Load models from S3
    rf_model = load_model_from_s3(bucket_name, rf_model_key)
    lof_model = load_model_from_s3(bucket_name, lof_model_key)

    # Fetch transactions data from Supabase
    transactions_data = fetch_transactions()

    if not transactions_data.empty:
        if st.button('Run Inference') and rf_model and lof_model:
            run_inference(transactions_data, rf_model, lof_model)

        # Configure and display the table using AgGrid
        gb = GridOptionsBuilder.from_dataframe(transactions_data)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)  # Set number of rows per page
        gb.configure_side_bar()  # Enable side bar
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
        grid_options = gb.build()
        AgGrid(transactions_data, gridOptions=grid_options, enable_enterprise_modules=True)
    else:
        st.write("No transactions data available.")

# Run this page function
transactions_page()
