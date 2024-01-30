import streamlit as st
import pandas as pd
import random

def offline_review_page():
    st.title('Offline Review')

    # Fetch unified flags and anomaly detection records from session state
    unified_flags = pd.DataFrame(st.session_state.get('unified_flags', []))
    anomaly_detection_records = pd.DataFrame(st.session_state.get('anomaly_detection_records', []))

    # Ensure 'ref_id' column is present in both dataframes
    if 'ref_id' in unified_flags.columns and 'ref_id' in anomaly_detection_records.columns:
        # Merge both dataframes on 'ref_id'
        review_transactions = pd.merge(unified_flags, anomaly_detection_records, on='ref_id', how='outer')
    else:
        st.error("Error: 'ref_id' column is missing in unified flags or anomaly detection records.")
        return

# Display transactions for review
st.subheader('Transactions for Offline Review')
if not review_transactions.empty:
    # Filter to show only fraud records
    fraud_transactions = review_transactions[review_transactions['flag_type'] == 'fraud']
    st.dataframe(fraud_transactions)

    # Check for required fields
    required_fields = ['customer_age', 'employment_status', 'housing_status']
    if not all(field in fraud_transactions.columns for field in required_fields):
        st.error('Required data for offline review is missing. Please ensure all necessary fields are included.')
        return  # Exit the function if required fields are missing

    # Button to simulate the offline review
    if st.button('Run Offline Review Simulation'):
        decisions = simulate_offline_review(fraud_transactions)
        st.write(decisions)  # For demonstration, simply print decisions
        st.success('Offline review simulation completed and decisions simulated.')
        st.session_state['offline_review_decisions'] = decisions
else:
    st.write("No transactions require offline review.")


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

