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
    """
    Load a machine learning model from an AWS S3 bucket.
    """
    aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
    aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
    model_str = response['Body'].read()
    with gzip.GzipFile(fileobj=io.BytesIO(model_str)) as file:
        model = pickle.load(file)
    st.write(f"Model {model_key} loaded successfully.")  # Debugging message
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
    """
    Preprocess transaction data for model inference.
    """
    df = df.drop(columns=['ref_id'], errors='ignore')
    categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    for col in categorical_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
    st.write("Data preprocessing completed.")  # Debugging message
    return df


def perform_inference(transactions_df, rf_model, lof_model):
    # Preprocess the data
    transactions_df = preprocess_data(transactions_df)

    # RF model predictions
    X = transactions_df.drop(['ref_id', 'fraud_bool'], axis=1, errors='ignore')  # Assuming 'ref_id' and 'fraud_bool' should not be included in features
    rf_predictions = rf_model.predict(X)
    rf_prob_scores = rf_model.predict_proba(X)[:, 1]
    
    # Add RF predictions and scores to DataFrame
    transactions_df['rf_prob_scores'] = rf_prob_scores
    transactions_df['rf_predicted_fraud'] = rf_predictions

    # Filter transactions predicted as non-fraud by RF for LOF processing
    non_fraud_df = transactions_df[transactions_df['rf_predicted_fraud'] == 0]

    # LOF model for anomaly detection on non-fraud transactions
    if not non_fraud_df.empty:
        X_lof = non_fraud_df.drop(['ref_id', 'fraud_bool', 'rf_predicted_fraud', 'rf_prob_scores'], axis=1, errors='ignore')
        lof_predictions = lof_model.fit_predict(X_lof)
        lof_scores = -lof_model.negative_outlier_factor_

        # Add LOF predictions and scores to non_fraud_df
        non_fraud_df['lof_predicted_fraud'] = (lof_predictions == -1).astype(int)
        non_fraud_df['lof_scores'] = lof_scores

        # Update the original DataFrame with LOF results
        transactions_df.update(non_fraud_df[['lof_predicted_fraud', 'lof_scores']])

    # Initialize LOF related columns in transactions_df if LOF was not applied
    if 'lof_predicted_fraud' not in transactions_df.columns:
        transactions_df['lof_predicted_fraud'] = 0
        transactions_df['lof_scores'] = 0

    # Normalize LOF scores if applicable
    if transactions_df['lof_scores'].max() != 0:  # Avoid division by zero
        transactions_df['lof_scores_normalized'] = (transactions_df['lof_scores'] - transactions_df['lof_scores'].min()) / (transactions_df['lof_scores'].max() - transactions_df['lof_scores'].min())
    else:
        transactions_df['lof_scores_normalized'] = 0

    # rf_v1_df contains all transactions with RF predictions
    rf_v1_df = transactions_df.copy()

    # lof_v1_df contains transactions flagged as anomalies by LOF
    lof_v1_df = transactions_df[transactions_df['lof_predicted_fraud'] == 1].copy()

    # Human review df contains transactions flagged as fraud by either model
    human_review_df = transactions_df[(transactions_df['rf_predicted_fraud'] == 1) | (transactions_df['lof_predicted_fraud'] == 1)].copy()

    return rf_v1_df, lof_v1_df, human_review_df

def app():
    st.title("Transaction Analysis")
    # Load models and fetch transactions as before
    rf_model = load_model_from_s3('your_bucket_name', 'rf_model_key')
    lof_model = load_model_from_s3('your_bucket_name', 'lof_model_key')
    transactions_df = fetch_transactions()  # Ensure this fetches and preprocesses the data
    
    if st.button('Fetch and Analyze Transactions'):
        if not transactions_df.empty:
            rf_v1_df, lof_v1_df, human_review_df = perform_inference(transactions_df, rf_model, lof_model)
            
            # Display RF_v1 predictions
            st.subheader("RF_v1 Supervised Predictions")
            st.dataframe(rf_v1_df)
            
            # Display LOF_v1 anomaly detection
            st.subheader("LOF_v1 Anomaly Detection")
            st.dataframe(lof_v1_df)
            
            # Display Human Review table
            st.subheader("Human Review (Fraud Predictions from RF_v1 and LOF_v1)")
            st.dataframe(human_review_df)
        else:
            st.write("No transactions found.")

if __name__ == '__main__':
    app()
