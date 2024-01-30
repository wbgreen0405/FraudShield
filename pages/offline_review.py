import streamlit as st
import pandas as pd
import random

import streamlit as st
import pandas as pd

def offline_review_page():
    st.title('Offline Review')

    # Fetch Unified Flags transactions from session state
    unified_flags_transactions = pd.DataFrame(st.session_state.get('unified_flags', []))

    # Fetch Anomaly Detection transactions from session state
    anomaly_detection_transactions = pd.DataFrame(st.session_state.get('anomaly_detection_records', []))

    # Display Unified Flags transactions
    st.subheader('Unified Flags for Review')
    if not unified_flags_transactions.empty:
        st.dataframe(unified_flags_transactions)
    else:
        st.write("No unified flag transactions available for review.")

    # Display Anomaly Detection transactions
    st.subheader('Anomaly Detection for Review')
    if not anomaly_detection_transactions.empty:
        st.dataframe(anomaly_detection_transactions)
    else:
        st.write("No anomaly detection transactions available for review.")


    
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

