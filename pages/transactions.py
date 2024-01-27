import streamlit as st
import pandas as pd
import pickle
import gzip
import zipfile
import io

# Helper function to load a model from a GZIP file
def load_model(uploaded_file):
    with gzip.open(uploaded_file, 'rb') as file:
        return pickle.load(file)

# Helper function to read transactions data from a ZIP file
def read_transactions_data(uploaded_file):
    with zipfile.ZipFile(uploaded_file, 'r') as zfile:
        # Assuming there is only one CSV file in the ZIP
        csv_filename = zfile.namelist()[0]
        with zfile.open(csv_filename) as csvfile:
            return pd.read_csv(csvfile)

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

def transactions_page():
    st.title('Transactions')

    # File uploaders for transaction data and models
    uploaded_transactions = st.file_uploader("Upload transactions data (ZIP file)", type=['zip'])
    uploaded_rf_model = st.file_uploader("Upload Random Forest model (GZIP file)", type=['gz'])
    uploaded_lof_model = st.file_uploader("Upload LOF model (GZIP file)", type=['gz'])

    # Conditional processing based on file uploads
    if uploaded_transactions and uploaded_rf_model and uploaded_lof_model:
        transactions_data = read_transactions_data(uploaded_transactions)
        rf_model = load_model(uploaded_rf_model)
        lof_model = load_model(uploaded_lof_model)

        if st.button('Run Inference'):
            run_inference(transactions_data, rf_model, lof_model)

        st.dataframe(transactions_data)
    else:
        st.write("Please upload all required files.")


# Run this page function
transactions_page()


