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
    """
    Perform inference on transaction data using RF and LOF models.
    """
    # Save 'ref_id' before dropping or excluding it from the feature set
    ref_ids = transactions_df['ref_id'].copy()
    
    # Preprocess data
    transactions_df = preprocess_data(transactions_df)
    
    # RF predictions
    X_rf = transactions_df.drop(['fraud_bool'], axis=1, errors='ignore')
    rf_predictions = rf_model.predict(X_rf)
    rf_prob_scores = rf_model.predict_proba(X_rf)[:, 1]  # Probability of being fraud
    transactions_df['rf_prob_scores'] = rf_prob_scores
    transactions_df['rf_predicted_fraud'] = rf_predictions

    # Initialize LOF columns
    transactions_df['lof_predicted_fraud'] = 0
    transactions_df['lof_scores'] = 0
    
    # Applying LOF on transactions classified as non-fraud by RF
    non_fraud_df = transactions_df[transactions_df['rf_predicted_fraud'] == 0].copy()
    if not non_fraud_df.empty:
        X_lof = non_fraud_df.drop(['fraud_bool', 'rf_predicted_fraud', 'rf_prob_scores'], axis=1, errors='ignore')
        lof_model.fit(X_lof)
        lof_predictions = lof_model.fit_predict(X_lof)
        lof_scores = -lof_model.negative_outlier_factor_  # Negative scores because higher means more abnormal
        
        # Assign LOF scores and predictions to the corresponding transactions
        transactions_df.loc[non_fraud_df.index, 'lof_predicted_fraud'] = lof_predictions
        transactions_df.loc[non_fraud_df.index, 'lof_scores'] = lof_scores

    # Normalize LOF scores for the whole dataset
    max_score = transactions_df['lof_scores'].max()
    min_score = transactions_df['lof_scores'].min()
    transactions_df['lof_scores_normalized'] = (transactions_df['lof_scores'] - min_score) / (max_score - min_score)

    # Assign back 'ref_id' and 'Approval Status'
    transactions_df['ref_id'] = ref_ids
    transactions_df['Approval Status'] = transactions_df['rf_predicted_fraud'].apply(lambda x: 'Fraud' if x == 1 else 'Non-Fraud')
    transactions_df['for_review'] = transactions_df.apply(lambda x: 1 if x['rf_predicted_fraud'] == 1 or x['lof_predicted_fraud'] == 1 else 0, axis=1)
    # After obtaining predictions from both models, set 'Approval Status' to 'Fraud' if either model predicts a case as fraud
    #transactions_df['Approval Status'] = transactions_df.apply(
        #lambda x: 'Fraud' if x['rf_predicted_fraud'] == 1 or x['lof_predicted_fraud'] == 1 else 'Non-Fraud', axis=1)

            
    return transactions_df

def app():
    st.title("Transaction Analysis")

    # Define your bucket name and model keys
    bucket_name = 'frauddetectpred'
    rf_model_key = 'random_forest_model.pkl.gz'
    lof_model_key = 'lof_nonfraud.pkl.gz'
    
    # Load models from S3
    rf_model = load_model_from_s3(bucket_name, rf_model_key)
    lof_model = load_model_from_s3(bucket_name, lof_model_key)

    if st.button('Fetch and Analyze Transactions'):
        # Assuming fetch_transactions() fetches and returns the transaction dataframe
        transactions_df = fetch_transactions()
        if not transactions_df.empty:
            analyzed_df = perform_inference(transactions_df, rf_model, lof_model)
            st.write("Analyzed Transactions:")
            st.dataframe(analyzed_df)
            st.session_state['analyzed_df'] = analyzed_df

            # Filtering for the Approval System and Anomaly Detection System as before
            supervised_df = analyzed_df[analyzed_df['rf_predicted_fraud'] == 1 | analyzed_df['rf_predicted_fraud'] == 0]
            st.dataframe(supervised_df)
            st.session_state['supervised_df'] = supervised_df

            anomaly_df = analyzed_df[analyzed_df['lof_predicted_fraud'] == 1]
            st.dataframe(anomaly_df)
            st.session_state['anomaly_df'] = anomaly_df

            # Offline Review Detailed Transactions filtering for fraud cases as predicted by RF or LOF
            review_df = analyzed_df[analyzed_df['for_review'] == 1]
            cols_order = ['ref_id', 'Approval Status', 'lof_scores', 'rf_prob_scores'] + [col for col in review_df.columns if col not in ['ref_id', 'Approval Status', 'lof_scores', 'rf_prob_scores']]
            review_df = review_df[cols_order]
            st.dataframe(review_df)
            st.session_state['review_df'] = review_df
        else:
            st.write("No transactions found.")

if __name__ == '__main__':
    st.set_page_config(page_title="Transaction Analysis", layout="wide")
    app()
