import streamlit as st
import pandas as pd
import random

def offline_review_page():
    st.title('Offline Review')

    # Check if the necessary data is in the session state
    if 'unified_flags' in st.session_state and 'anomaly_detection_records' in st.session_state:
        unified_flags_df = pd.DataFrame(st.session_state['unified_flags'])
        anomaly_detection_df = pd.DataFrame(st.session_state['anomaly_detection_records'])

        # Merge unified flags and anomaly detection records
        review_transactions = pd.merge(unified_flags_df, anomaly_detection_df, on='ref_id', how='outer', suffixes=('_rf', '_lof'))

        # Display transactions to review
        st.subheader('Transactions for Offline Review')
        if not review_transactions.empty:
            st.dataframe(review_transactions)
            # ... [rest of your offline review simulation code] ...
        else:
            st.write("No transactions require offline review.")
    else:
        st.write("Data for offline review is not available. Please run the inference first.")
    
def simulate_offline_review(transaction_data):
    # Define your thresholds and suspicious criteria
    INCOME_THRESHOLD = 100000
    AGE_THRESHOLD = 50
    EMPLOYMENT_STATUS_SUSPICIOUS = 3  # Adjust based on your encoding
    HOUSING_STATUS_SUSPICIOUS = 2     # Adjust based on your encoding
    ERROR_RATE = 0.1

    decisions = {}
    for index, transaction in transaction_data.iterrows():
        is_unusually_high_income = transaction['flag_reason'] > INCOME_THRESHOLD
        is_age_above_threshold = transaction['customer_age'] > AGE_THRESHOLD
        is_suspicious_employment = transaction['employment_status'] == EMPLOYMENT_STATUS_SUSPICIOUS
        is_suspicious_housing = transaction['housing_status'] == HOUSING_STATUS_SUSPICIOUS

        if is_unusually_high_income and (is_age_above_threshold or is_suspicious_employment or is_suspicious_housing):
            decision = 'fraudulent'
        else:
            decision = 'legitimate'

        if random.random() < ERROR_RATE:
            decision = 'legitimate' if decision == 'fraudulent' else 'fraudulent'

        decisions[transaction['ref_id']] = decision
    return decisions

# Run this page function
offline_review_page()

