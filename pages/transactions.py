import streamlit as st
import numpy as np
import pandas as pd
import pickle
import boto3
import gzip
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from supabase import create_client, Client

# Set the option
#pd.set_option('future.no silent downcasting', True)

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
    # Drop 'ref_id' column if it exists to avoid errors during processing
    #df = df.drop(columns=['ref_id'], errors='ignore')
    categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    for col in categorical_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
    return df

def preprocess_data(df):
    """
    Placeholder for the actual preprocessing steps.
    Adjust according to your actual preprocessing requirements.
    """
    # Example preprocessing steps
    categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    for col in categorical_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
    return df

def perform_inference(transactions_df, rf_model, lof_model):
    # Initialize 'LOF Status' column to ensure it's always present
    transactions_df['LOF Status'] = 'Not Evaluated'
    
    # Save 'ref_id' before preprocessing if it exists, create a placeholder if it doesn't
    ref_ids = transactions_df['ref_id'].copy() if 'ref_id' in transactions_df.columns else pd.Series(range(len(transactions_df)), name='ref_id')
    
    # Preprocess the data, excluding 'ref_id'
    transactions_df = transactions_df.drop(columns=['ref_id'], errors='ignore')
    transactions_df = preprocess_data(transactions_df)

    # Exclude 'LOF Status' and any other non-numeric or post-processing columns for RF predictions
    X_rf = transactions_df.drop(columns=['fraud_bool', 'LOF Status'], errors='ignore')
    rf_prob_scores = rf_model.predict_proba(X_rf)[:, 1]
    rf_predictions = [1 if prob > 0.5 else 0 for prob in rf_prob_scores]

    # Update DataFrame with RF predictions and scores
    transactions_df['rf_prob_scores'] = rf_prob_scores
    transactions_df['rf_predicted_fraud'] = rf_predictions
    transactions_df['RF Approval Status'] = transactions_df['rf_predicted_fraud'].map({1: 'Marked as Fraud', 0: 'Marked as Approve'})

    # Apply LOF on transactions classified as non-fraud by RF
    non_fraud_df = transactions_df[transactions_df['rf_predicted_fraud'] == 0].copy()

    # Reattach 'ref_id' for merging purposes
    non_fraud_df['ref_id'] = ref_ids[non_fraud_df.index]

    if not non_fraud_df.empty:
        X_lof = non_fraud_df.drop(columns=['fraud_bool', 'rf_predicted_fraud', 'rf_prob_scores', 'RF Approval Status', 'ref_id', 'LOF Status'], errors='ignore')
        lof_predictions = lof_model.fit_predict(X_lof)
        lof_scores = -lof_model.negative_outlier_factor_
        st.write("LOF scores generated:", lof_scores[:5])  # Print first 5 scores for inspection

        # Map LOF predictions and scores back to non_fraud_df
        non_fraud_df['LOF Status'] = pd.Series(lof_predictions, index=non_fraud_df.index).map({-1: 'Suspected Fraud', 1: 'Non-Fraud'})
        non_fraud_df['lof_scores'] = lof_scores


        # Update the main DataFrame with LOF results
        transactions_df.update(non_fraud_df)

    # Normalize LOF scores if present
    if 'lof_scores' in transactions_df.columns:
        max_score = transactions_df['lof_scores'].max()
        min_score = transactions_df['lof_scores'].min()
        transactions_df['lof_scores_normalized'] = (transactions_df['lof_scores'] - min_score) / (max_score - min_score) if max_score > min_score else 0

    # Reattach 'ref_id' after all processing
    transactions_df['ref_id'] = ref_ids.values
 

    return transactions_df, non_fraud_df


def app():
    st.title("Transaction Analysis")

    # Using specified bucket name and keys
    bucket_name = 'frauddetectpred'
    rf_model_key = 'random_forest_model.pkl.gz'
    lof_model_key = 'lof_nonfraud.pkl.gz'
    
    try:
        # Load the Random Forest and LOF models from S3
        rf_model = load_model_from_s3(bucket_name, rf_model_key)
        lof_model = load_model_from_s3(bucket_name, lof_model_key)
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return

    if st.button('Fetch and Analyze Transactions'):
        transactions_df = fetch_transactions()  # Placeholder for actual data fetching logic
        if not transactions_df.empty:
            #analyzed_df = perform_inference(transactions_df, rf_model, lof_model)
            analyzed_df, non_fraud_df = perform_inference(transactions_df, rf_model, lof_model)

            # Make sure that lof_scores are in non_fraud_df
            if 'lof_scores' not in non_fraud_df.columns:
                st.error("LOF scores are missing in non_fraud_df.")
            return
            
            # Update transactions_df with non_fraud_df information
            transactions_df = transactions_df.merge(non_fraud_df[['ref_id', 'lof_scores', 'LOF Status']], on='ref_id', how='left')

               
            st.write("Analyzed Transactions:")
            st.dataframe(analyzed_df)
            st.session_state['analyzed_df'] = analyzed_df

            # Debugging: Check unique values in the 'RF Approval Status' and 'LOF Status' columns
            st.write("Unique RF Approval Status values:", analyzed_df['RF Approval Status'].unique())
            st.write("Unique LOF Status values:", analyzed_df['LOF Status'].unique())

            # Debugging: Check DataFrame after calculating LOF scores
            st.write("DataFrame after LOF score calculation:", non_fraud_df.head())
            
            # After updating the main DataFrame
            transactions_df.update(non_fraud_df)
            st.write("Updated main DataFrame with LOF scores:", transactions_df.head())


            # Filter based on RF Approval Status and LOF Status
            supervised_df = analyzed_df[(analyzed_df['RF Approval Status'] == 'Marked as Fraud') | (analyzed_df['RF Approval Status'] == 'Marked as Approve')]
            st.write("### Approval System")
            st.dataframe(supervised_df)
            st.session_state['supervised_df'] = supervised_df
    
            #non_fraud_df = analyzed_df[analyzed_df['LOF Status'] == 'Suspected Fraud']
            st.write("### Anomaly Detection System")
            st.dataframe(non_fraud_df)
            st.session_state['anomaly_df'] = non_fraud_df


            if 'lof_scores' not in analyzed_df.columns:
                analyzed_df['lof_scores'] = np.nan
            # Prepare Offline Review Detailed Transactions with merged flags
            analyzed_df['Flagged By'] = np.where(analyzed_df['RF Approval Status'] == 'Marked as Fraud', 'RF Model', 
                                                 np.where(analyzed_df['LOF Status'] == 'Suspected Fraud', 'LOF Model', 'None'))
            review_df = analyzed_df[(analyzed_df['RF Approval Status'] == 'Marked as Fraud') | (analyzed_df['LOF Status'] == 'Suspected Fraud')]
            cols_order = ['ref_id', 'Flagged By', 'RF Approval Status', 'LOF Status', 'lof_scores', 'rf_prob_scores'] + [col for col in analyzed_df.columns if col not in ['ref_id', 'Flagged By', 'RF Approval Status', 'LOF Status', 'lof_scores', 'rf_prob_scores']]
            review_df = review_df[cols_order]
            st.write("### Offline Review Detailed Transactions")
            st.dataframe(review_df)
            st.session_state['review_df'] = review_df

            # Additional debugging output as before
            st.write("### Debugging: Filtered DataFrames")
            st.write("Original Data Shape:", transactions_df.shape)
            st.write("Approval System Shape:", supervised_df.shape)
            st.write(supervised_df['RF Approval Status'].value_counts())
            st.write("Anomaly Detection System Shape:", non_fraud_df .shape)
            st.write(non_fraud_df['LOF Status'].value_counts())
            st.write("Offline Review Shape:", review_df.shape)
            st.write(review_df[['RF Approval Status', 'LOF Status']].value_counts())
        else:
            st.write("No transactions found.")

if __name__ == '__main__':
    st.set_page_config(page_title="Transaction Analysis", layout="wide")
    app()





