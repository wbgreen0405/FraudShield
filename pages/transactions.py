import streamlit as st
import pandas as pd
import pickle
import boto3
import gzip
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from supabase import create_client, Client

# Initialize Supabase client using Streamlit secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase: Client = create_client(supabase_url, supabase_key)

def load_model_from_s3(bucket_name, model_key):
    """Load a machine learning model from an AWS S3 bucket."""
    aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
    aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
    model_str = response['Body'].read()
    with gzip.GzipFile(fileobj=io.BytesIO(model_str)) as file:
        model = pickle.load(file)
    return model

def fetch_transactions():
    """Fetch transactions data from a Supabase table."""
    try:
        response = supabase.table('transactions').select('*').execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.error(f'An error occurred: {e}')
        return pd.DataFrame()

def preprocess_data(df):
    """Preprocess transaction data for model inference."""
    df = df.drop(columns=['ref_id'], errors='ignore')
    categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def perform_inference(transactions_df, rf_model, lof_model):
    """Perform inference on transaction data using RF and LOF models."""
    transactions_df = preprocess_data(transactions_df)
    
    # RF predictions
    X_rf = transactions_df.drop(['fraud_bool'], axis=1, errors='ignore')
    rf_predictions = rf_model.predict(X_rf)
    transactions_df['rf_predicted_fraud'] = rf_predictions
    rf_probabilities = rf_model.predict_proba(X_rf)[:, 1]  # Assuming a binary classification task
    
    # Initialize LOF predictions column to all zeros
    transactions_df['lof_predicted_fraud'] = 0
    non_fraud_df = transactions_df[transactions_df['rf_predicted_fraud'] == 0].copy()
    
    if not non_fraud_df.empty:
        X_lof = non_fraud_df.drop(['fraud_bool', 'rf_predicted_fraud'], axis=1, errors='ignore')
        lof_predictions = lof_model.fit_predict(X_lof)
        lof_scores = lof_model.negative_outlier_factor_
        non_fraud_df['lof_predicted_fraud'] = (lof_predictions == -1).astype(int)
        transactions_df.update(non_fraud_df[['lof_predicted_fraud']])
    
    # Prepare review_transactions_info for offline review detailed DataFrame
    review_transactions_info = [
        {'transaction_index': i, 'model_type': 'Random Forest', 'score': prob}
        for i, prob in enumerate(rf_probabilities) if transactions_df.iloc[i]['rf_predicted_fraud'] == 1
    ]
    for i, score in enumerate(lof_scores):
        if transactions_df.iloc[i]['lof_predicted_fraud'] == 1:
            entry = {'transaction_index': i, 'model_type': 'LOF', 'score': score}
            review_transactions_info.append(entry)
    
    df_offline_review_detailed = pd.DataFrame(review_transactions_info)
    df_offline_review_detailed = df_offline_review_detailed.merge(transactions_df.reset_index(), left_on='transaction_index', right_on='index', how='left').drop('index', axis=1)
    
    # Storing DataFrames in session state for cross-page access
    st.session_state['approval_system_df'] = transactions_df
    st.session_state['anomaly_detection_system_df'] = transactions_df[transactions_df['lof_predicted_fraud'] == 1]
    st.session_state['df_offline_review_detailed'] = df_offline_review_detailed

    return transactions_df

import streamlit as st

def app():
    st.title("Transaction Analysis")

    # Using specified bucket name and keys
    bucket_name = 'frauddetectpred'
    rf_model_key = 'random_forest_model.pkl.gz'
    lof_model_key = 'lof_nonfraud.pkl.gz'
    
    # Load the Random Forest and LOF models from S3
    rf_model = load_model_from_s3(bucket_name, rf_model_key)
    lof_model = load_model_from_s3(bucket_name, lof_model_key)

    if st.button('Fetch and Analyze Transactions'):
        transactions_df = fetch_transactions()
        if not transactions_df.empty:
            # Perform inference and ensure the 'ref_id' column is used as the identifier
            analyzed_df = perform_inference(transactions_df, rf_model, lof_model)

            # Display Analyzed Transactions with proper column names
            if 'ref_id' in analyzed_df and 'rf_predicted_fraud' in analyzed_df:
                analyzed_df['Fraud Status'] = analyzed_df['rf_predicted_fraud'].apply(lambda x: 'Fraud' if x == 1 else 'Not Fraud')
                st.write("### Analyzed Transactions:")
                st.dataframe(analyzed_df[['ref_id', 'Fraud Status']])

                # Anomaly Detection System
                st.write("### Anomaly Detection System")
                if 'lof_predicted_fraud' in analyzed_df:
                    anomaly_df = analyzed_df[analyzed_df['lof_predicted_fraud'] == 1]
                    st.dataframe(anomaly_df[['ref_id', 'Fraud Status']])
                    st.session_state['anomaly_df'] = anomaly_df
                else:
                    st.write("LOF predictions are not available.")

                # Offline Review Detailed Transactions
                st.write("### Offline Review Detailed Transactions")
                review_df = analyzed_df[(analyzed_df['rf_predicted_fraud'] == 1) | (analyzed_df['lof_predicted_fraud'] == 1)]
                st.dataframe(review_df[['ref_id', 'Fraud Status']])
                st.session_state['review_df'] = review_df
            else:
                st.error("Required columns for display are missing.")
        else:
            st.write("No transactions found.")

if __name__ == '__main__':
    st.set_page_config(page_title="Transaction Analysis", layout="wide")
    app()



