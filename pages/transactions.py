import streamlit as st
import pandas as pd
import pickle
import boto3
import gzip
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from supabase import create_client

# Initialize Supabase client using Streamlit secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase = create_client(supabase_url, supabase_key)

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
    st.write(f"Model {model_key} loaded successfully.")
    return model

def fetch_transactions():
    """
    Fetch transactions from Supabase.
    """
    response = supabase.table('transactions').select('*').execute()
    if response.error:
        st.error(f'Failed to retrieve data. Error: {response.error.message}')
        return pd.DataFrame()
    return pd.DataFrame(response.data)

def preprocess_data(df):
    """
    Preprocess the transaction data.
    """
    df = df.drop(columns=['ref_id'], errors='ignore')
    categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    for col in categorical_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
    st.write("Data preprocessing completed.")
    return df

def perform_inference(transactions_df, rf_model, lof_model):
    """
    Perform inference using RF and LOF models.
    """
    ref_ids = transactions_df['ref_id'].copy()
    transactions_df = preprocess_data(transactions_df)
    
    # RF predictions
    X_rf = transactions_df.drop(columns=['fraud_bool'], errors='ignore')
    rf_predictions = rf_model.predict(X_rf)
    rf_prob_scores = rf_model.predict_proba(X_rf)[:, 1]
    transactions_df['rf_prob_scores'] = rf_prob_scores
    transactions_df['rf_predicted_fraud'] = rf_predictions

    # Apply LOF model only to non-fraud by RF
    non_fraud_df = transactions_df[transactions_df['rf_predicted_fraud'] == 0].copy()
    if not non_fraud_df.empty:
        X_lof = non_fraud_df.drop(columns=['fraud_bool', 'rf_predicted_fraud', 'rf_prob_scores'], errors='ignore')
        lof_model.fit(X_lof)
        lof_predictions = lof_model.fit_predict(X_lof)
        lof_scores = -lof_model.negative_outlier_factor_
        transactions_df.loc[non_fraud_df.index, 'lof_predicted_fraud'] = (lof_predictions == -1).astype(int)
        transactions_df.loc[non_fraud_df.index, 'lof_scores'] = lof_scores

    transactions_df['ref_id'] = ref_ids
    transactions_df['Approval Status'] = transactions_df.apply(
        lambda x: 'Fraud' if x['rf_predicted_fraud'] == 1 or x['lof_predicted_fraud'] == 1 else 'Non-Fraud', axis=1)

    # Creating separate tables for review
    rf_v1_df = transactions_df.copy()
    lof_v1_df = transactions_df[transactions_df['lof_predicted_fraud'] == 1].copy()
    human_review_df = transactions_df[(transactions_df['rf_predicted_fraud'] == 1) | (transactions_df['lof_predicted_fraud'] == 1)].copy()

    return rf_v1_df, lof_v1_df, human_review_df

def app():
    st.title("Transaction Analysis")

    bucket_name = 'frauddetectpred'
    rf_model_key = 'random_forest_model.pkl.gz'
    lof_model_key = 'lof_nonfraud.pkl.gz''
    
    rf_model = load_model_from_s3(bucket_name, rf_model_key)
    lof_model = load_model_from_s3(bucket_name, lof_model_key)

    if st.button('Fetch and Analyze Transactions'):
        transactions_df = fetch_transactions()
        if not transactions_df.empty:
            rf_v1_df, lof_v1_df, human_review_df = perform_inference(transactions_df, rf_model, lof_model)
            
            st.write("### Approval System (RF_v1 Predictions)")
            st.dataframe(rf_v1_df[['ref_id', 'rf_prob_scores', 'rf_predicted_fraud', 'Approval Status']])

            st.write("### Anomaly Detection System (LOF_v1 Anomalies)")
            st.dataframe(lof_v1_df[['ref_id', 'lof_scores', 'lof_predicted_fraud']])

            st.write("### Offline Review (Detailed Transactions for Human Review)")
            st.dataframe(human_review_df[['ref_id', 'rf_prob_scores', 'lof_scores', 'Approval Status']])
        else:
            st.write("No transactions found.")

if __name__ == '__main__':
    st.set_page_config(page_title="Transaction Analysis", layout="wide")
    app()
