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
#def fetch_transactions():
    #try:
        #response = supabase.table('transactions').select('*').limit(100).execute()
        # Check for errors in the response
        #if hasattr(response, 'error') and response.error:
            #st.error(f'Failed to retrieve data. Error: {str(response.error)}')
            #return pd.DataFrame()
        #elif hasattr(response, 'data'):
            #return pd.DataFrame(response.data)
        #else:
            #st.error('Unexpected response format.')
            #return pd.DataFrame()
    #except Exception as e:
        #st.error(f'An error occurred: {e}')
        #return pd.DataFrame()

#def fetch_transactions():
    #try:
        #response = supabase.table('transactions').select('*').execute()
        # Log the entire response for debugging
        #st.write("Response received:", response)

        # Check for errors in the response
        #if hasattr(response, 'error') and response.error:
            #st.error(f'Failed to retrieve data. Error: {str(response.error)}')
            #return pd.DataFrame()
        #elif hasattr(response, 'data'):
            #if response.data:
                #return pd.DataFrame(response.data)
            #else:
                #st.warning('No data found in the transactions table.')
                #return pd.DataFrame()
        #else:
            #st.error('Unexpected response format.')
            #return pd.DataFrame()
    #except Exception as e:
        #st.error(f'An error occurred: {e}')
        #return pd.DataFrame()

#def fetch_transactions():
    #try:
        #response = supabase.table('transactions').select('ref_id').execute()  # Assuming 'id' is a column in your table
        #st.write("Response received:", response)

        #if hasattr(response, 'error') and response.error:
            #st.error(f'Failed to retrieve data. Error: {str(response.error)}')
            #return pd.DataFrame()
        #elif hasattr(response, 'data'):
            #if response.data:
                #return pd.DataFrame(response.data)
            #else:
                #st.warning('No data found in the transactions table.')
                #return pd.DataFrame()
        #else:
            #st.error('Unexpected response format.')
            #return pd.DataFrame()
    #except Exception as e:
        #st.error(f'An error occurred: {e}')
        #return pd.DataFrame()

#def fetch_transactions():
    #try:
        #response = supabase.table('transactions').select('*').limit(100).execute()
        
        # Check for errors in the response
        #if hasattr(response, 'error') and response.error:
            #st.error(f'Failed to retrieve data. Error: {str(response.error)}')
            #return None  # Return None or appropriate placeholder if there's an error
       # elif hasattr(response, 'data'):
            #return response.data  # Return the raw data
        #else:
            #st.error('Unexpected response format.')
            #return None
    #except Exception as e:
        #st.error(f'An error occurred: {e}')
        #return None


#def fetch_transactions():
    #try:
        #response = supabase.table('transactions').select('ref_id').execute()

        # Checking if response is successful
        #if response.status_code == 200:
            #data = response.data
            #if data:
                #return pd.DataFrame(data)
            #else:
                #st.write("Transactions table is empty.")
                #return pd.DataFrame()
        #else:
            #st.error(f'Request failed with status code: {response.status_code}')
            #return pd.DataFrame()

    #except Exception as e:
        #st.error(f'An error occurred: {e}')
        #return pd.DataFrame()

#def fetch_transactions():
    #try:
        # Fetch only 100 rows from the 'transactions' table
        #data, error = supabase.table('transactions').select('*').limit(100).execute()

        # Check if there's an error in the response
        #if error:
            #st.error(f'Failed to retrieve data. Error: {error.message}')
            #return pd.DataFrame()

        #return pd.DataFrame(data)
    #except Exception as e:
        #st.error(f'An error occurred: {e}')
        #return pd.DataFrame()

def fetch_transactions():
    try:
        response = supabase.table('transactions').select('*').limit(100).execute()
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




def run_inference(transactions_data):
    # Load models
    with open(RF_MODEL_PATH, 'rb') as file:
        rf_model = pickle.load(file)
        # Store the model in the session state so it can be accessed from other pages
        st.session_state['rf_model'] = rf_model
        
    with open(LOF_MODEL_PATH, 'rb') as file:
        lof_model = pickle.load(file)

    # Predict potential fraud cases with probabilities
    rf_probabilities = rf_model.predict_proba(transactions_data)[:, 1]
    rf_predictions = [1 if prob > 0.5 else 0 for prob in rf_probabilities]

    # Filter out transactions flagged as potential fraud and non-fraud
    potential_fraud_indices = [i for i, pred in enumerate(rf_predictions) if pred == 1]
    potential_nonfraud_indices = [i for i, pred in enumerate(rf_predictions) if pred == 0]
    X_potential_nonfraud = transactions_data.iloc[potential_nonfraud_indices]

    # Apply LOF model on potential non-fraud cases
    lof_anomaly_indices = []
    if len(X_potential_nonfraud) > 20:
        lof_predictions = lof_model.fit_predict(X_potential_nonfraud)
        lof_anomaly_indices = [index for index, pred in zip(potential_nonfraud_indices, lof_predictions) if pred == -1]

    # Combine LOF anomalies and RF frauds for human review
    offline_review_transactions = set(potential_fraud_indices + lof_anomaly_indices)

    # Prepare data for saving
    save_unified_flags(transactions_data.iloc[potential_fraud_indices], rf_predictions, rf_probabilities)
    save_anomaly_detection_records(transactions_data.iloc[lof_anomaly_indices], lof_anomaly_indices)

    st.session_state['rf_predictions'] = rf_predictions
    st.session_state['rf_probabilities'] = rf_probabilities
    st.session_state['potential_fraud_indices'] = potential_fraud_indices
    st.session_state['lof_anomaly_indices'] = lof_anomaly_indices

    st.success("Inference complete and results saved.")

#def transactions_page():
    #st.title('Transactions')

    # Load models from uploaded files
    #uploaded_rf_model = st.file_uploader("Upload Random Forest model (GZIP file)", type=['gz'])
    #uploaded_lof_model = st.file_uploader("Upload LOF model (GZIP file)", type=['gz'])
    #if uploaded_rf_model and uploaded_lof_model:
        #rf_model = load_model(uploaded_rf_model)
        #lof_model = load_model(uploaded_lof_model)
    #else:
        #st.write("Please upload model files to run inference.")

    # Fetch transactions data from Supabase
    #transactions_data = fetch_transactions()

    #if not transactions_data.empty:
        #if st.button('Run Inference'):
            #run_inference(transactions_data, rf_model, lof_model)
        #st.dataframe(transactions_data)
    #else:
       #st.write("No transactions data available.")

#def transactions_page():
    #st.title('Transactions')

    # Load models from uploaded files
    #uploaded_rf_model = st.file_uploader("Upload Random Forest model (GZIP file)", type=['gz'])
    #uploaded_lof_model = st.file_uploader("Upload LOF model (GZIP file)", type=['gz'])

    #rf_model, lof_model = None, None
    #if uploaded_rf_model and uploaded_lof_model:
        #rf_model = load_model(uploaded_rf_model)
        #lof_model = load_model(uploaded_lof_model)
    #else:
        #st.write("Please upload model files to run inference.")

    # Fetch transactions data from Supabase
    #transactions_data = fetch_transactions()

    #if transactions_data and not transactions_data.empty:
        #if st.button('Run Inference'):
            #run_inference(transactions_data, rf_model, lof_model)
        #st.dataframe(transactions_data)
    #else:
        #st.write("No transactions data available or an error occurred.")

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

    # Corrected boolean check for DataFrame
    if not transactions_data.empty:
        if st.button('Run Inference'):
            run_inference(transactions_data, rf_model, lof_model)
        st.dataframe(transactions_data)
    else:
        st.write("No transactions data available.")


# Run this page function
transactions_page()



