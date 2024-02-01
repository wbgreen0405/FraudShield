import streamlit as st
import pandas as pd
from supabase import create_client, Client

# Initialize Supabase client using Streamlit secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase: Client = create_client(supabase_url, supabase_key)

def supervised_fraud_results_page():
    st.title('Supervised Fraud Model Results')

    # Fetch model results from Supabase tables
    rf_confusion_matrix = fetch_supabase_table("rf_confusion_matrix")
    rf_feature_importance = fetch_supabase_table("rf_feature_importance")
    rf_model_metrics = fetch_supabase_table("rf_model_metrics")

    # Display the fetched results
    st.subheader("Random Forest Confusion Matrix:")
    st.dataframe(rf_confusion_matrix)

    st.subheader("Random Forest Feature Importance:")
    st.dataframe(rf_feature_importance)

    st.subheader("Random Forest Model Metrics:")
    st.dataframe(rf_model_metrics)

# Helper function to fetch data from Supabase tables
def fetch_supabase_table(table_name):
    try:
        response = supabase.table(table_name).select('*').execute()
        if hasattr(response, 'error') and response.error:
            st.error(f'Failed to retrieve data from {table_name}. Error: {str(response.error)}')
            return pd.DataFrame()
        elif hasattr(response, 'data'):
            return pd.DataFrame(response.data)
        else:
            st.error(f'Unexpected response format from {table_name}.')
            return pd.DataFrame()
    except Exception as e:
        st.error(f'An error occurred while fetching data from {table_name}: {e}')
        return pd.DataFrame()



