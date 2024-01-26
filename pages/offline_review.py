import streamlit as st
import pandas as pd
from utils.supabase_ops import fetch_offline_review_transactions, save_offline_review_decisions

def offline_review_page():
    st.title('Offline Review')

    # Fetch transactions that require offline review
    review_transactions = fetch_offline_review_transactions()

    # If there are transactions to review
    if not review_transactions.empty:
        # Display transactions to review
        st.subheader('Transactions for Offline Review')
        st.dataframe(review_transactions)

        # Button to simulate the offline review
        if st.button('Run Offline Review Simulation'):
            decisions = simulate_offline_review(review_transactions)
            save_offline_review_decisions(decisions)
            st.success('Offline review simulation completed and decisions saved.')
    else:
        st.write("No transactions require offline review.")

def simulate_offline_review(transaction_data):
    # Define your thresholds and suspicious criteria
    INCOME_THRESHOLD = 100000
    AGE_THRESHOLD = 50
    EMPLOYMENT_STATUS_SUSPICIOUS = 3
    HOUSING_STATUS_SUSPICIOUS = 2
    ERROR_RATE = 0.1

    decisions = {}
    for index, transaction in transaction_data.iterrows():
        is_unusually_high_income = transaction['income'] > INCOME_THRESHOLD
        is_age_above_threshold = transaction['customer_age'] > AGE_THRESHOLD
        is_suspicious_employment = transaction['employment_status'] == EMPLOYMENT_STATUS_SUSPICIOUS
        is_suspicious_housing = transaction['housing_status'] == HOUSING_STATUS_SUSPICIOUS

        if is_unusually_high_income and (is_age_above_threshold or is_suspicious_employment or is_suspicious_housing):
            decision = 'fraudulent'
        else:
            decision = 'legitimate'

        # Simulate random errors in decision making
        if random.random() < ERROR_RATE:
            decision = 'legitimate' if decision == 'fraudulent' else 'fraudulent'

        decisions[index] = decision
    return decisions

# Run this page function
offline_review_page()
