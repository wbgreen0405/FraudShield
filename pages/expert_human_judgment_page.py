import streamlit as st
import pandas as pd
import datetime
import random
from pages import audit_logs_history_page  # Import the audit_logs_history_page function

# Define an empty list to store audit logs
audit_logs = []

# Function to log an entry in the audit logs
def log_audit_entry(transaction_id, reviewer_id, decision):
    timestamp = datetime.datetime.now()
    audit_entry = {
        'Timestamp': timestamp,
        'Transaction ID': transaction_id,
        'Reviewer ID': reviewer_id,
        'Decision': decision,
    }
    audit_logs.append(audit_entry)

def expert_human_judgment_page():
    st.title("Expert Human Judgment")

    # Access the offline_review_transactions from session_state
    offline_review_transactions = st.session_state.get('offline_review_transactions', [])

    if not offline_review_transactions:
        st.info("No transactions available for expert human judgment.")
        return

    st.write("Here you can simulate the expert human judgment process.")

    # Function to simulate offline review
    def simulate_offline_review(transaction_data, flagged_indices):
        INCOME_THRESHOLD = 100000
        AGE_THRESHOLD = 50
        EMPLOYMENT_STATUS_SUSPICIOUS = 3
        HOUSING_STATUS_SUSPICIOUS = 2
        ERROR_RATE = 0.1

        decisions = {}
        for index in flagged_indices:
            transaction = transaction_data.iloc[index]
            is_unusually_high_income = transaction['income'] > INCOME_THRESHOLD
            is_age_above_threshold = transaction['customer_age'] > AGE_THRESHOLD
            is_suspicious_employment = transaction['employment_status'] == EMPLOYMENT_STATUS_SUSPICIOUS
            is_suspicious_housing = transaction['housing_status'] == HOUSING_STATUS_SUSPICIOUS

            if is_unusually_high_income and (is_age_above_threshold or is_suspicious_employment or is_suspicious_housing):
                decision = 'fraudulent'
            else:
                decision = 'legitimate'

            if random.random() < ERROR_RATE:
                decision = 'legitimate' if decision == 'fraudulent' else 'fraudulent'

            decisions[index] = decision

            # Log an entry in the audit logs for this decision
            log_audit_entry(transaction_id=index, reviewer_id='simulated_reviewer', decision=decision)

        return decisions

    # Retrieve transactions_data from session_state
    transactions_data = st.session_state.get('transactions_data', None)

    if transactions_data is None:
        st.error("Transactions data is missing. Please run preprocessing and inference first.")
        return

    # Simulate offline review for flagged transactions
    offline_review_decisions = simulate_offline_review(transactions_data, offline_review_transactions)

    # Prepare data for Human Review Table
    human_review_records = []
    for index, decision in offline_review_decisions.items():
        human_review_records.append({
            'review_id': index,  # Assuming index as review_id for simplicity
            'ref_id': index,
            'reviewed_at': datetime.datetime.now(),
            'reviewer_id': 'simulated_reviewer',
            'decision': decision,
            'comments': 'Simulated review decision'
        })

    # Create a DataFrame to display the simulated human review decisions
    human_review_df = pd.DataFrame(human_review_records)

    # Display the human review decisions in a table
    st.write("Simulated Human Review Decisions:")
    st.write(human_review_df)

    # Call the audit_logs_history_page function (without parentheses) to display audit logs
    audit_logs_history_page.audit_logs_history_page(audit_logs)

if __name__ == '__main__':
    expert_human_judgment_page()


