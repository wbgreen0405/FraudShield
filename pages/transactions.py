import streamlit as st
import pandas as pd
import pickle
from st_aggrid import AgGrid
from datetime import datetime
from utils.supabase_ops import fetch_transactions, save_unified_flags, save_anomaly_detection_records

# File paths for the saved models
RF_MODEL_PATH = 'path/to/random_forest_model.pkl'
LOF_MODEL_PATH = 'path/to/lof_nonfraud.pkl'

def run_inference(transactions_data):
    # Load models
    with open(RF_MODEL_PATH, 'rb') as file:
        rf_model = pickle.load(file)
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
    save_unified_flags(transactions_data, rf_predictions, rf_probabilities)
    save_anomaly_detection_records(transactions_data, lof_anomaly_indices)

    st.write("Inference complete and results saved.")

def transactions_page():
    st.title('Transactions')
    transactions_data = fetch_transactions()

    if st.button('Run Inference'):
        run_inference(transactions_data)

    if not transactions_data.empty:
        AgGrid(transactions_data)
    else:
        st.write("No transactions data to display.")


