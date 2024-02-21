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
    if 'mappings' not in st.session_state:
        st.session_state['mappings'] = {}
    
    categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    for col in categorical_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            # Save the mapping from class to label for later use
            st.session_state['mappings'][col] = {i: label for i, label in enumerate(encoder.classes_)}
    
    return df



# Example function to preprocess data and save mappings
def preprocess_data_and_save_mappings(df):
    categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    mappings = {}
    for col in categorical_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            # Fit and transform the data
            df[col] = encoder.fit_transform(df[col].astype(str))
            # Save the mapping from labels to integers
            mappings[col] = {index: label for index, label in enumerate(encoder.classes_)}
    # Save mappings to session state
    st.session_state['mappings'] = mappings
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

def app():
    st.title("Transaction Analysis")

    # Using specified bucket name and keys
    bucket_name = 'frauddetectpred'
    rf_model_key = 'random_forest_model.pkl.gz'
    lof_model_key = 'lof_nonfraud.pkl.gz'
    
    # Only attempt to load models and fetch transactions if analysis hasn't been performed
    if 'analysis_performed' not in st.session_state:
        try:
            # Load the Random Forest and LOF models from S3
            rf_model = load_model_from_s3(bucket_name, rf_model_key)
            lof_model = load_model_from_s3(bucket_name, lof_model_key)

            # Fetch transactions and perform analysis
            transactions_df = fetch_transactions()
            if transactions_df.empty:
                st.error("No transactions found.")
                return

            analyzed_df, non_fraud_df = perform_inference(transactions_df, rf_model, lof_model)
            if 'lof_scores' not in non_fraud_df.columns:
                st.error("LOF scores are missing in non_fraud_df.")
                return

            # Merge non_fraud_df information into analyzed_df
            analyzed_df = analyzed_df.merge(non_fraud_df[['ref_id', 'lof_scores', 'LOF Status']], on='ref_id', how='left')

            st.session_state['analyzed_df'] = analyzed_df

            supervised_df = analyzed_df[(analyzed_df['RF Approval Status'] == 'Marked as Fraud') | (analyzed_df['RF Approval Status'] == 'Marked as Approve')]
            st.session_state['supervised_df'] = supervised_df

            st.session_state['anomaly_df'] = non_fraud_df

            # Update 'analyzed_df' with merged information
            analyzed_df = analyzed_df.merge(
                non_fraud_df[['ref_id', 'lof_scores', 'LOF Status']],
                on='ref_id',
                how='left',
                suffixes=('', '_y')
            )

            st.session_state['review_df'] = analyzed_df

            # Flag transactions for review based on RF Approval Status and LOF Status
            analyzed_df['flag_for_review'] = (
                (analyzed_df['RF Approval Status'] == 'Marked as Fraud') | 
                (analyzed_df['LOF Status'] == 'Suspected Fraud')
            )
            
            # Filter the DataFrame to include only transactions flagged for review
            case_review_df = analyzed_df[analyzed_df['flag_for_review']]


            # Save the merged DataFrame for case review
            st.session_state['case_review_df'] =  case_review_df


            # Set analysis performed flag
            st.session_state['analysis_performed'] = True
            # Indicate that transaction analysis is completed successfully
            st.session_state['transaction_analysis_completed'] = True

        except Exception as e:
            st.error(f"Failed to load models or process transactions: {e}")
            return

    if 'analysis_performed' in st.session_state and st.session_state['analysis_performed']:
        # Display results or further actions if analysis has been performed
        col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
        with col_metrics1:
            st.metric("Total Transactions Analyzed", len(st.session_state['analyzed_df']))
        with col_metrics2:
            st.metric("Transactions Flagged for Review by RF", len(st.session_state['analyzed_df'][st.session_state['analyzed_df']['RF Approval Status'] == 'Marked as Fraud']))
        with col_metrics3:
            st.metric("Transactions Flagged for Review by LOF", len(st.session_state['anomaly_df'][st.session_state['anomaly_df']['LOF Status'] == 'Suspected Fraud']))
        with col_metrics4:
            st.metric("Transactions Flagged for Offline Review", len(st.session_state['review_df'][st.session_state['review_df']['RF Approval Status'] == 'Marked as Fraud']) + len(st.session_state['anomaly_df'][st.session_state['anomaly_df']['LOF Status'] == 'Suspected Fraud']))

        # Begin adding visualizations in columns after your existing code
        data = st.session_state['case_review_df']  # Load case review data

            
        col_viz1, col_viz2 = st.columns(2)  # Create two columns for visualizations

        with col_viz1:
            # Reverse mapping for 'payment_type'
            if 'payment_type' in st.session_state['mappings']:
                reverse_mapping_payment_type = {v: k for k, v in st.session_state['mappings']['payment_type'].items()}
                data['payment_type'] = data['payment_type'].map(reverse_mapping_payment_type)
            
            fig_payment_type = px.bar(data, x='payment_type', title='Applications by Payment Type',
                                      color='payment_type', 
                                      labels={'payment_type': 'Payment Type'})
            st.plotly_chart(fig_payment_type)

        with col_viz2:
            # Reverse mapping for 'employment_status'
            if 'employment_status' in st.session_state['mappings']:
                reverse_mapping_employment_status = {v: k for k, v in st.session_state['mappings']['employment_status'].items()}
                data['employment_status'] = data['employment_status'].map(reverse_mapping_employment_status)
            
            fig_employment_status = px.bar(data, x='employment_status', title='Employment Status Distribution',
                                           color='employment_status', 
                                           labels={'employment_status': 'Employment Status'})
            st.plotly_chart(fig_employment_status)

            # Reverse mapping for 'housing_status'
            if 'housing_status' in st.session_state['mappings']:
                reverse_mapping_housing_status = {v: k for k, v in st.session_state['mappings']['housing_status'].items()}
                data['housing_status'] = data['housing_status'].map(reverse_mapping_housing_status)
            
            fig_housing_status = px.bar(data, x='housing_status', title='Housing Status Distribution',
                                        color='housing_status', 
                                        labels={'housing_status': 'Housing Status'})
            st.plotly_chart(fig_housing_status)

if __name__ == "__main__":
    app()

