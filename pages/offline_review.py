import streamlit as st
import pandas as pd
import random

def offline_review_page():
    st.title('Offline Review')

    # Fetch transactions that require offline review from session state
    if 'unified_flags' in st.session_state:
        review_transactions = pd.DataFrame(st.session_state['unified_flags'])
    else:
        review_transactions = pd.DataFrame()

    # Check for required columns in review_transactions
    required_columns = ['ref_id', 'customer_age', 'employment_status', 'housing_status', 'flag_reason']
    if not all(col in review_transactions.columns for col in required_columns):
        st.error("Required data for offline review is missing. Please ensure all necessary fields are included.")
        return

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

