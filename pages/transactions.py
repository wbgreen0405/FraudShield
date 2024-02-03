import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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

def fetch_transactions():
    # Function to fetch transactions from Supabase
    response = supabase.table("transactions").select("*").execute()
    if response.error:
        st.error(f"Failed to fetch transactions: {response.error}")
        return pd.DataFrame()
    return pd.DataFrame(response.data)

def preprocess_data(df):
    # Data preprocessing logic
    df = df.copy()
    # Example preprocessing steps
    for col in ['column_a', 'column_b']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            df[col] = LabelEncoder().fit_transform(df[col])
    return df

def run_inference(transactions_data, rf_model, lof_model, selected_features):
    # Placeholder for your model inference logic
    # Assuming your models return a DataFrame with a 'fraud_probability' column
    preprocessed_data = preprocess_data(transactions_data[selected_features])
    rf_predictions = rf_model.predict(preprocessed_data)
    lof_predictions = lof_model.predict(preprocessed_data)
    # Example logic for determining combined flags indices
    potential_fraud_indices = rf_predictions[rf_predictions['fraud_probability'] > 0.5].index.tolist()
    lof_anomaly_indices = lof_predictions[lof_predictions == -1].index.tolist()
    combined_flags_indices = list(set(potential_fraud_indices) | set(lof_anomaly_indices))
    return combined_flags_indices

def create_combined_flags_table(combined_flags_indices, transactions_data, selected_features):
    # Creating a combined flags table
    combined_flags_data = transactions_data.loc[combined_flags_indices]
    combined_flags_data['flag'] = 'Potential Fraud/Anomaly'
    return combined_flags_data

def display_transactions_data(transactions_data):
    # Using AgGrid to display transactions data
    gb = GridOptionsBuilder.from_dataframe(transactions_data)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    grid_options = gb.build()
    AgGrid(transactions_data, gridOptions=grid_options, enable_enterprise_modules=True)

def display_combined_data(unified_flags, anomaly_detection_records):
    # Displaying unified flags and anomaly detection records if available
    if unified_flags or anomaly_detection_records:
        st.write("Combined Flags and Anomaly Detection Table:")
        combined_table = pd.concat([pd.DataFrame(unified_flags), pd.DataFrame(anomaly_detection_records)], ignore_index=True)
        st.dataframe(combined_table)

def transactions_page():
    st.set_page_config(layout="wide")
    st.title('Transactions')

    # Load models and fetch transactions
    bucket_name = 'frauddetectpred'
    rf_model = load_model_from_s3(bucket_name, 'random_forest_model.pkl.gz')
    lof_model = load_model_from_s3(bucket_name, 'lof_nonfraud.pkl.gz')

    transactions_data = fetch_transactions()

    if transactions_data.empty:
        st.error("No transactions data available.")
        return

    if st.button('Run Preprocessing and Inference'):
        with st.spinner('Running preprocessing and inference...'):
            selected_features = st.session_state.get('selected_features', transactions_data.columns.tolist())
            combined_flags_indices = run_inference(transactions_data, rf_model, lof_model, selected_features)
            
            combined_flags_table = create_combined_flags_table(combined_flags_indices, transactions_data, selected_features)
            
            if not combined_flags_table.empty:
                st.session_state['combined_flags_table'] = combined_flags_table
                st.write("Combined Flags Table:")
                st.dataframe(combined_flags_table)
            else:
                st.error("Failed to create a valid DataFrame for combined_flags_table.")
    else:
        st.info("Click the button to run preprocessing and inference.")

    display_transactions_data(transactions_data)

    # Display unified flags and anomaly detection records (assuming they are stored in session_state)
    unified_flags = st.session_state.get('unified_flags', [])
    anomaly_detection_records = st.session_state.get('anomaly_detection_records', [])
    display_combined_data(unified_flags, anomaly_detection_records)

if __name__ == '__main__':
    transactions_page()

