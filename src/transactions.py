import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
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
    #st.write(f"Model {model_key} loaded successfully.")  # Debugging message
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
        #st.write("LOF scores generated:", lof_scores[:5])  # Print first 5 scores for inspection

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

def create_visualizations(fraud_df):
    """
    Create visualizations for the confirmed fraudulent transactions.
    """

    # Payment Type and Credit Limit Analysis
    col1, col2 = st.columns(2)
    with col1:
        # Fraudulent Transactions by Payment Type
        fig_payment_type = px.histogram(fraud_df, x='payment_type', title='Fraudulent Transactions by Payment Type')
        st.plotly_chart(fig_payment_type)
    
    with col2:
        # Credit Limit Requests in Confirmed Fraudulent Transactions
        fig_credit_limit = px.box(fraud_df, y='proposed_credit_limit', title='Credit Limit Requests in Confirmed Fraudulent Transactions')
        st.plotly_chart(fig_credit_limit)

    # Customer Demographics and Application Details
    col3, col4, col5 = st.columns([1, 1, 1])
    with col3:
        # Age and Employment Status Distribution
        fig_age_employment = px.histogram(fraud_df, x='customer_age', color='employment_status', title='Age and Employment Status in Fraudulent Transactions')
        st.plotly_chart(fig_age_employment)

    with col4:
        # Housing Status Comparison
        fig_housing_status = px.histogram(fraud_df, x='housing_status', title='Housing Status in Fraudulent Transactions')
        st.plotly_chart(fig_housing_status)

    # Assuming 'email_is_free' and 'phone_mobile_valid' as binary represented as True/False
    with col5:
        # Email and Phone Number Validity
        fig_contact_info = px.histogram(fraud_df, x=['email_is_free', 'phone_mobile_valid'], title='Contact Information Validity')
        st.plotly_chart(fig_contact_info)

    # Bank Branch and Zip Code Activity
    col6, col7 = st.columns(2)
    with col6:
        # Applications per Bank Branch
        fig_bank_branch = px.histogram(fraud_df, x='bank_branch_count_8w', title='Applications per Bank Branch in Fraudulent Transactions')
        st.plotly_chart(fig_bank_branch)

    with col7:
        # Zip Code Application Density
        fig_zip_code_density = px.histogram(fraud_df, x='zip_count_4w', title='Zip Code Application Density in Fraudulent Transactions')
        st.plotly_chart(fig_zip_code_density)

    # Add additional visualizations as needed based on the provided data points and definitions


# Main app function
def app():
    st.title("Transaction Analysis")

    # Your initialization of Supabase client and setting up bucket and model keys
    supabase_url = st.secrets["supabase"]["url"]
    supabase_key = st.secrets["supabase"]["key"]
    supabase = create_client(supabase_url, supabase_key)
    bucket_name = 'frauddetectpred'
    rf_model_key = 'random_forest_model.pkl.gz'
    lof_model_key = 'lof_nonfraud.pkl.gz'

    # Load models and perform analysis if not done already
    if 'analysis_performed' not in st.session_state:
        try:
            rf_model = load_model_from_s3(bucket_name, rf_model_key)
            lof_model = load_model_from_s3(bucket_name, lof_model_key)
            transactions_df = fetch_transactions()
            if transactions_df.empty:
                st.error("No transactions found.")
                return
            analyzed_df, non_fraud_df = perform_inference(transactions_df, rf_model, lof_model)
            st.session_state['analyzed_df'] = analyzed_df
            st.session_state['supervised_df'] = analyzed_df[(analyzed_df['RF Approval Status'] == 'Marked as Fraud') | (analyzed_df['LOF Status'] == 'Suspected Fraud')]
            st.session_state['anomaly_df'] = non_fraud_df
            st.session_state['review_df'] = analyzed_df
            st.session_state['analysis_performed'] = True
        except Exception as e:
            st.error(f"Error in analysis: {e}")
            return

    # Display results and visualizations if analysis has been performed
    if 'analysis_performed' in st.session_state:
        fraud_df = st.session_state['analyzed_df'][(st.session_state['analyzed_df']['RF Approval Status'] == 'Marked as Fraud') | (st.session_state['analyzed_df']['LOF Status'] == 'Suspected Fraud')]
        if not fraud_df.empty:
            create_visualizations(fraud_df)
        else:
            st.write("No confirmed fraud cases to display.")

    if 'analysis_performed' in st.session_state and st.session_state['analysis_performed']:
        # Display results or further actions if analysis has been performed
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions Analyzed", len(st.session_state['analyzed_df']))
        with col2:
            st.metric("Transactions Flagged for Review by RF", len(st.session_state['analyzed_df'][st.session_state['analyzed_df']['RF Approval Status'] == 'Marked as Fraud']))
        with col3:
            st.metric("Transactions Flagged for Review by LOF", len(st.session_state['anomaly_df'][st.session_state['anomaly_df']['LOF Status'] == 'Suspected Fraud']))
        with col4:
            st.metric("Transactions Flagged for Offline Review", len(st.session_state['review_df'][st.session_state['review_df']['RF Approval Status'] == 'Marked as Fraud']) + len(st.session_state['anomaly_df'][st.session_state['anomaly_df']['LOF Status'] == 'Suspected Fraud']))
        # This ensures that the final statement aligns with the block it's supposed to be part of


if __name__ == "__main__":
    app()

