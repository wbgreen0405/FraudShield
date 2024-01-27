import streamlit as st
import pandas as pd
import pickle
import gzip
from supabase import create_client, Client

# Initialize Supabase client using Streamlit secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase: Client = create_client(supabase_url, supabase_key)

# Helper function to load a model from a GZIP file
def load_model(uploaded_file):
    with gzip.open(uploaded_file, 'rb') as file:
        return pickle.load(file)

# Function to fetch transactions from Supabase
def fetch_transactions():
    try:
        data = supabase.table('transactions').select('*').limit(100).execute()
        if data.error:
            st.error(f'Failed to retrieve data. Error: {data.error.message}')
            return pd.DataFrame()
        return pd.DataFrame(data.data)
    except Exception as e:
        st.error(f'An error occurred: {e}')
        return pd.DataFrame()

def run_inference(transactions_data, rf_model, lof_model):
    # Your existing inference logic here...

def transactions_page():
    st.title('Transactions')

    # Load models from uploaded files
    uploaded_rf_model = st.file_uploader("Upload Random Forest model (GZIP file)", type=['gz'])
    uploaded_lof_model = st.file_uploader("Upload LOF model (GZIP file)", type=['gz'])
    if uploaded_rf_model and uploaded_lof_model:
        rf_model = load_model(uploaded_rf_model)
        lof_model = load_model(uploaded_lof_model)
    else:
        st.write("Please upload model files to run inference.")

    # Fetch transactions data from Supabase
    transactions_data = fetch_transactions()

    if not transactions_data.empty:
        if st.button('Run Inference'):
            run_inference(transactions_data, rf_model, lof_model)
        st.dataframe(transactions_data)
    else:
        st.write("No transactions data available.")

# Run this page function
transactions_page()



