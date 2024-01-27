import streamlit as st
from .config import supabase
import pandas as pd
from datetime import datetime


from supabase import create_client, Client

# Initialize Supabase client using Streamlit secrets
# Initialize Supabase client using Streamlit secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase: Client = create_client(supabase_url, supabase_key)

#def fetch_transactions():
    # Logic to fetch transactions from Supabase
    #try:
        #data, error = supabase.table('transactions').select('*').execute()
        
        # Check if there's an error in the response
        #if error:
            #st.error(f'Failed to retrieve data. Error: {error}')
            #return pd.DataFrame()
        #else:
            #return pd.DataFrame(data)

    #except Exception as e:
        #st.error(f'An error occurred: {e}')
        #return pd.DataFrame()

def fetch_transactions():
    try:
        response = supabase.table('transactions').select('*').execute()
        print("Response:", response)  # Debug print
        data = response.data
        if data:
            print("Data:", data)  # Debug print
            return pd.DataFrame(data)
        else:
            print("No data was fetched. Error:", response.error)  # Debug print
            st.error('Failed to retrieve data')
            return pd.DataFrame()
    except Exception as e:
        print("An exception occurred:", e)  # Debug print
        st.error(f'An error occurred: {e}')
        return pd.DataFrame()



def save_unified_flags(transactions_data, rf_predictions, rf_probabilities):
    unified_flags = []
    for index, (pred, prob) in enumerate(zip(rf_predictions, rf_probabilities)):
        if pred == 1:
            unified_flags.append({
                'flag_id': transactions_data.iloc[index]['transaction_id'],  # assuming 'transaction_id' is a column in your table
                'ref_id': transactions_data.iloc[index]['transaction_id'],
                'flagged_at': datetime.now(),
                'model_version': 'RF_v1',
                'flag_reason': prob,
                'flag_type': 'fraud'
            })
    # Insert unified_flags into Supabase
    response = supabase.table('unified_flags_table').insert(unified_flags).execute()
    if response.error:
        st.error(f"Failed to save unified flags: {response.error.message}")

def save_anomaly_detection_records(transactions_data, lof_anomaly_indices, lof_nonfraud_scores, lof_threshold_nonfraud):
    anomaly_detection_records = []
    for index in lof_anomaly_indices:
        anomaly_detection_records.append({
            'anomaly_id': transactions_data.iloc[index]['transaction_id'],
            'ref_id': transactions_data.iloc[index]['transaction_id'],
            'checked_at': datetime.now(),
            'anomaly_score': lof_nonfraud_scores[index],
            'threshold': lof_threshold_nonfraud
        })
    # Insert anomaly_detection_records into Supabase
    response = supabase.table('anomaly_detection_table').insert(anomaly_detection_records).execute()
    if response.error:
        st.error(f"Failed to save anomaly detection records: {response.error.message}")

def fetch_unified_flags():
    # Fetch unified flags from Supabase
    data = supabase.table('unified_flags').select('*').execute()
    if data.error:
        st.error(f"Failed to fetch unified flags: {data.error.message}")
        return pd.DataFrame()
    return pd.DataFrame(data.data)

def update_unified_flag(flag_id, status):
    # Update the status of a unified flag in Supabase
    response = supabase.table('unified_flags').update({'status': status}).eq('id', flag_id).execute()
    if response.error:
        st.error(f"Failed to update flag: {response.error.message}")


#def fetch_offline_review_transactions():
    #try:
        # Attempt to fetch data from the table
        #response = supabase.table('offline_review_transactions').select('*').execute()

        # Check if the response is successful
        #if response.status_code == 200:
            #return pd.DataFrame(response.data)
        #else:
            #st.warning('Offline review transactions table does not exist yet.')
            #return pd.DataFrame()

   # except Exception as e:
        #st.error(f'An error occurred while fetching offline review transactions: {e}')
       # return pd.DataFrame()

def fetch_offline_review_transactions():
    # Temporarily bypassing this function as the table doesn't exist yet
    # You can add the relevant code here once the table is created
    return pd.DataFrame()


def save_offline_review_decisions(decisions):
    # Save the offline review decisions to Supabase
    for index, decision in decisions.items():
        response = supabase.table('offline_review_decisions').update({'decision': decision}).eq('transaction_id', index).execute()
        if response.error:
            st.error(f"Failed to save offline review decision for transaction {index}: {response.error.message}")
