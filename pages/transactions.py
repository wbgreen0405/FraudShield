import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier  # Assuming use of RF
from sklearn.neighbors import LocalOutlierFactor  # Assuming use of LOF for anomaly detection
import boto3
import gzip
import io
from supabase import create_client, Client

# Initialize Supabase client using Streamlit secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase: Client = create_client(supabase_url, supabase_key)

def load_model_from_s3(bucket_name, model_key):
    """
    Load a machine learning model from an AWS S3 bucket.
    """
    aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
    aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
    model_str = response['Body'].read()
    with gzip.GzipFile(fileobj=io.BytesIO(model_str)) as file:
        model = pickle.load(file)
    return model

def fetch_transactions():
    """
    Fetch transactions data from a Supabase table.
    """
    try:
        response = supabase.table('transactions').select('*').execute()
        if response.error:
            st.error(f'Failed to retrieve data. Error: {str(response.error)}')
            return pd.DataFrame()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f'An error occurred: {e}')
        return pd.DataFrame()

def preprocess_data(df):
    """
    Preprocess transaction data for model inference.
    """
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

def perform_inference(transactions_df, rf_model, lof_model):
    """
    Perform inference on transaction data using RF and LOF models.
    """
    transactions_df = preprocess_data(transactions_df)
    
    # RF predictions
    X_rf = transactions_df.drop(['fraud_bool'], axis=1, errors='ignore')
    rf_predictions = rf_model.predict(X_rf)
    transactions_df['rf_predicted_fraud'] = rf_predictions

    # Applying LOF on transactions classified as non-fraud by RF
    non_fraud_df = transactions_df[transactions_df['rf_predicted_fraud'] == 0].copy()
    if not non_fraud_df.empty:
        X_lof = non_fraud_df.drop(['fraud_bool', 'rf_predicted_fraud'], axis=1, errors='ignore')
        lof_predictions = lof_model.fit_predict(X_lof)
        non_fraud_df['lof_predicted_fraud'] = (lof_predictions == -1).astype(int)
        transactions_df.update(non_fraud_df[['lof_predicted_fraud']])

    return transactions_df

def app():
    st.title("Transaction Analysis")

    # Load models
    bucket_name = 'your_bucket_name'  # Update with your actual bucket name
    rf_model = load_model_from_s3(bucket_name, 'random_forest_model.pkl.gz')
    lof_model = load_model_from_s3(bucket_name, 'lof_model.pkl.gz')

    if st.button('Fetch and Analyze Transactions'):
        transactions_df = fetch_transactions()
        if not transactions_df.empty:
            analyzed_df = perform_inference(transactions_df, rf_model, lof_model)
            
            # Display Analyzed Transactions
            st.write("Analyzed Transactions:")
            st.dataframe(analyzed_df)
            
            # Display Anomaly Detection System Transactions
            st.write("### Anomaly Detection System")
            anomaly_df = analyzed_df[analyzed_df['lof_predicted_fraud'] == 1]
            st.dataframe(anomaly_df)

            # Display Transactions for Offline Review (combining RF and LOF findings)
            st.write("### Offline Review Detailed Transactions")
            review_df = analyzed_df[(analyzed_df['rf_predicted_fraud'] == 1) | (analyzed_df['lof_predicted_fraud'] == 1)]
            st.dataframe(review_df)

            # Set session state to navigate to the Approval System page after processing
            st.session_state['navigate_to_approval'] = True
            st.experimental_rerun()  # Rerun the app to reflect the updated state
        else:
            st.write("No transactions found.")
