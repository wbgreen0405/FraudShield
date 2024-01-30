import streamlit as st
import pandas as pd
import random

def offline_review_page():
    st.title('Offline Review')

    # Fetch transactions that require offline review from session state
    review_transactions = pd.DataFrame()
    if 'unified_flags' in st.session_state:
        unified_flags_df = pd.DataFrame(st.session_state['unified_flags'])
        review_transactions = pd.concat([review_transactions, unified_flags_df], ignore_index=True)
    
    if 'anomaly_detection_records' in st.session_state:
        anomaly_records_df = pd.DataFrame(st.session_state['anomaly_detection_records'])
        review_transactions = pd.concat([review_transactions, anomaly_records_df], ignore_index=True)

    # If there are transactions to review
    if not review_transactions.empty:
        # Display transactions to review
        st.subheader('Transactions for Offline Review')
        st.dataframe(review_transactions)

        # Button to simulate the offline review
        if st.button('Run Offline Review Simulation'):
            decisions = simulate_offline_review(review_transactions)
            # For demonstration, simply print decisions
            st.write(decisions)
            st.success('Offline review simulation completed and decisions simulated.')

            # Optionally, store the decisions in the session state or elsewhere
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
        is_unusually_high_income = transaction['flag_reason'] > INCOME_THRESHOLD  # Assuming 'flag_reason' is income
        is_age_above_threshold = transaction['customer_age'] > AGE_THRESHOLD  # Make sure 'customer_age' is included
        is_suspicious_employment = transaction['employment_status'] == EMPLOYMENT_STATUS_SUSPICIOUS
        is_suspicious_housing = transaction['housing_status'] == HOUSING_STATUS_SUSPICIOUS

        if is_unusually_high_income and (is_age_above_threshold or is_suspicious_employment or is_suspicious_housing):
            decision = 'fraudulent'
        else:
            decision = 'legitimate'

        # Simulate random errors in decision making
        if random.random() < ERROR_RATE:
            decision = 'legitimate' if decision == 'fraudulent' else 'fraudulent'

        decisions[transaction['ref_id']] = decision
    return decisions

# Run this page function
offline_review_page()
