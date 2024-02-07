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
    Perform inference on transaction data using RF and LOF models, and flag the source.
    """
    # Save 'ref_id' before modifying the DataFrame
    ref_ids = transactions_df['ref_id'].copy()

    # Preprocess data
    transactions_df = preprocess_data(transactions_df)

    # Initialize a new column for tracking the source of fraud flag
    transactions_df['flagged_by'] = None

    # RF predictions
    X_rf = transactions_df.drop(['fraud_bool', 'ref_id'], axis=1, errors='ignore')
    rf_predictions = rf_model.predict(X_rf)
    rf_prob_scores = rf_model.predict_proba(X_rf)[:, 1]  # Probability of being fraud
    transactions_df['rf_prob_scores'] = rf_prob_scores
    transactions_df['rf_predicted_fraud'] = rf_predictions

    # Mark transactions flagged by RF
    transactions_df.loc[transactions_df['rf_predicted_fraud'] == 1, 'flagged_by'] = 'rf_v1'

    # Initialize LOF predictions and scores
    transactions_df['lof_predicted_fraud'] = 0
    transactions_df['lof_scores'] = 0

    # Applying LOF on transactions classified as non-fraud by RF
    non_fraud_df = transactions_df[transactions_df['rf_predicted_fraud'] == 0].copy()
    if not non_fraud_df.empty:
        X_lof = non_fraud_df.drop(['fraud_bool', 'rf_predicted_fraud', 'rf_prob_scores', 'ref_id'], axis=1, errors='ignore')
        lof_predictions = lof_model.fit_predict(X_lof)
        lof_scores = -lof_model.negative_outlier_factor_
        non_fraud_df['lof_scores'] = lof_scores
        non_fraud_df['lof_predicted_fraud'] = (lof_predictions == -1).astype(int)

        # Update the main DataFrame with LOF predictions and scores
        transactions_df.update(non_fraud_df)

        # Update flagged_by based on LOF predictions
        lof_fraud_indices = non_fraud_df[non_fraud_df['lof_predicted_fraud'] == 1].index
        transactions_df.loc[lof_fraud_indices, 'flagged_by'] = 'lof_v1'

    # Normalize LOF scores for the whole dataset
    transactions_df['lof_scores_normalized'] = (transactions_df['lof_scores'] - transactions_df['lof_scores'].min()) / (transactions_df['lof_scores'].max() - transactions_df['lof_scores'].min())

    # Assign back 'ref_id'
    transactions_df['ref_id'] = ref_ids

    # Update Approval Status based on combined predictions
    transactions_df['Approval Status'] = transactions_df.apply(
        lambda x: 'Non-Fraud' if x['rf_predicted_fraud'] == 0 and x['lof_predicted_fraud'] == 0 else 'Fraud', axis=1)

    # Optionally store DataFrames in session state for cross-page access
    # st.session_state['approval_system_df'] = transactions_df
    # st.session_state['anomaly_detection_system_df'] = transactions_df[transactions_df['lof_predicted_fraud'] == 1]
    # st.session_state['df_offline_review_detailed'] = transactions_df[(transactions_df['rf_predicted_fraud'] == 1) | (transactions_df['lof_predicted_fraud'] == 1)]

    return transactions_df

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
            analyzed_df = perform_inference(transactions_df, rf_model, lof_model)
            
            # Display sections as requested
            st.write("Analyzed Transactions:")
            st.dataframe(analyzed_df)
            st.session_state['analyzed_df'] = analyzed_df

            st.write("### Aproval System")
            supervised_df = analyzed_df[(analyzed_df['rf_predicted_fraud'] == 1) | (analyzed_df['rf_predicted_fraud'] == 0)]
            #st.dataframe(supervised_df[['ref_id', 'rf_predicted_fraud', 'Approval Status']])
            st.dataframe(supervised_df)
            st.session_state['supervised_df'] = supervised_df

            st.write("### Anomaly Detection System")
            anomaly_df = analyzed_df[analyzed_df['lof_predicted_fraud'] == 1]
            st.dataframe(anomaly_df)
            st.session_state['anomaly_df'] = anomaly_df

            st.write("### Offline Review Detailed Transactions")
            review_df = analyzed_df[(analyzed_df['rf_predicted_fraud'] == 1) | (analyzed_df['lof_predicted_fraud'] == 1)]
            st.dataframe(review_df)
            st.session_state['review_df'] = review_df
        else:
            st.write("No transactions found.")

if __name__ == '__main__':
    st.set_page_config(page_title="Transaction Analysis", layout="wide")
    app()
