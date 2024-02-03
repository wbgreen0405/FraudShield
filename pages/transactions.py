import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
import boto3
import pickle
import gzip
import io
from st_aggrid import AgGrid, GridOptionsBuilder
from supabase import create_client, Client

# Initialize Supabase client using Streamlit secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase = create_client(supabase_url, supabase_key)

def load_model_from_s3(bucket_name, model_key):
    # Function to load a model from S3
    s3 = boto3.client('s3', aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
                      aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"])
    response = s3.get_object(Bucket=bucket_name, Key=model_key)
    model_str = response['Body'].read()
    model = pickle.loads(gzip.decompress(model_str))
    return model

def load_lof_model_from_s3(bucket_name, model_key, novelty=False):
    # Function to load the LOF model from S3 with or without novelty detection
    s3 = boto3.client('s3', aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
                      aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"])
    response = s3.get_object(Bucket=bucket_name, Key=model_key)
    model_str = response['Body'].read()
    
    if novelty:
        # Load LOF model with novelty detection
        lof_model = pickle.loads(gzip.decompress(model_str))
    else:
        # Load LOF model without novelty detection
        lof_model = pickle.loads(gzip.decompress(model_str))
    
    return lof_model

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
    # Data preprocessing logic
    df = df.copy()
    for col in ['payment_type', 'source', 'device_os', 'employment_status', 'housing_status']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            df[col] = LabelEncoder().fit_transform(df[col])
    return df

def run_inference(transactions_data, rf_model, lof_model):
    # Store 'ref_id' in a separate Series to preserve the order
    ref_ids = transactions_data['ref_id'].copy()
    
    # Preprocess the data, excluding 'ref_id' for the prediction
    preprocessed_data = preprocess_data(transactions_data.drop(columns=['ref_id']))
    
    # Retrieve user-defined settings
    fraud_threshold = st.session_state.get('fraud_threshold', 0.5)  # Default to 0.5 if not set
    
    # Predict potential fraud cases with probabilities
    rf_probabilities = rf_model.predict_proba(preprocessed_data)[:, 1]
    rf_predictions = [1 if prob > fraud_threshold else 0 for prob in rf_probabilities]

    # Filter out transactions flagged as potential fraud and non-fraud
    potential_fraud_indices = [i for i, pred in enumerate(rf_predictions) if pred == 1]
    potential_nonfraud_indices = [i for i, pred in enumerate(rf_predictions) if pred == 0]
    X_potential_nonfraud = preprocessed_data.iloc[potential_nonfraud_indices]

    # Apply LOF model on potential non-fraud cases to further screen for anomalies
    # When initializing the LOF model
    lof_model = LocalOutlierFactor(n_neighbors=20, contamination='auto')
    lof_predictions = lof_model.predict(X_potential_nonfraud)
    
    # Identify indices of anomalies detected by LOF within the non-fraud cases
    lof_anomaly_indices = [potential_nonfraud_indices[i] for i, pred in enumerate(lof_predictions) if pred == -1]

    # Combine indices for review or further action
    combined_review_indices = set(potential_fraud_indices + lof_anomaly_indices)

    # Create a DataFrame to return, including the 'ref_id' for each index in combined_review_indices
    result_df = pd.DataFrame({'ref_id': ref_ids.iloc[list(combined_review_indices)].values, 'review_flag': 1})
    
    return result_df

def transactions_page():
    st.set_page_config(layout="wide")
    st.title('Transactions')

    # Load models and fetch transactions
    bucket_name = 'frauddetectpred'
    rf_model_key = 'random_forest_model.pkl.gz'
    lof_model_key = 'lof_nonfraud.pkl.gz'
    rf_model = load_model_from_s3(bucket_name, rf_model_key)
    lof_model = load_model_from_s3(bucket_name, lof_model_key)


    transactions_data = fetch_transactions()

    if not transactions_data.empty:
        display_transactions_data(transactions_data)
        
        if st.button('Run Preprocessing and Inference'):
            with st.spinner('Running preprocessing and inference...'):
                review_df = run_inference(transactions_data, rf_model, lof_model)  # Adjusted call with 3 arguments
                
                if not review_df.empty:
                    # This assumes you want to display the review DataFrame as is
                    st.write("Transactions Flagged for Review:")
                    st.dataframe(review_df)
                    
                    # Optional: Join review_df with transactions_data for detailed view
                    detailed_review = transactions_data.merge(review_df, on='ref_id', how='right')
                    st.write("Detailed Review Table:")
                    st.dataframe(detailed_review)
                else:
                    st.write("No transactions flagged for review.")

def display_transactions_data(transactions_data):
    # Using AgGrid to display transactions data
    gb = GridOptionsBuilder.from_dataframe(transactions_data)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    grid_options = gb.build()
    AgGrid(transactions_data, gridOptions=grid_options, enable_enterprise_modules=True)

if __name__ == '__main__':
    transactions_page()
