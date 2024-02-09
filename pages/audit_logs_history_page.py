import streamlit as st
import pandas as pd
from datetime import datetime

def log_audit_entry(transaction_id, reviewer_id, decision, audit_logs):
    timestamp = datetime.now()
    audit_entry = {
        'Timestamp': timestamp,
        'Transaction ID': transaction_id,
        'Reviewer ID': reviewer_id,
        'Decision': decision,
    }
    audit_logs.append(audit_entry)

def load_audit_logs_data(audit_logs):
    # Create a DataFrame from the passed audit_logs list
    audit_logs_df = pd.DataFrame(audit_logs)
    return audit_logs_df

def audit_logs_history_page():
    st.title("Audit Logs / History")
    st.write("View detailed transaction audit logs and the history of changes made through the UI.")

    # Initialize an empty list to store audit logs if not already in session state
    if 'audit_logs' not in st.session_state:
        st.session_state['audit_logs'] = []

    # Check if there is review data in session_state and log actions
    if 'review_df' in st.session_state and st.session_state['review_df'] is not None:
        review_df = st.session_state['review_df']
        for _, row in review_df.iterrows():
            log_audit_entry(
                transaction_id=row['ref_id'],
                reviewer_id="Expert Reviewer",
                decision=row.get('expert_decision', 'No Decision'),
                audit_logs=st.session_state['audit_logs']
            )

    # Load audit logs data
    audit_logs_df = load_audit_logs_data(st.session_state['audit_logs'])

    # Display audit logs
    st.subheader("Transaction Audit Logs")
    if not audit_logs_df.empty:
        st.dataframe(audit_logs_df)  # Display audit logs as a dataframe
    else:
        st.info("No audit logs available.")

# This is just a placeholder function for loading change history, which should be replaced with your actual data source logic.
def load_change_history_data():
    # Placeholder function for demonstration
    change_history_data = {
        'Timestamp': [datetime.now(), datetime.now()],
        'User': ['User A', 'User B'],
        'Change': ['Decision Updated', 'Decision Confirmed'],
        'Details': ['Decision: Confirmed Fraud', 'Decision: Confirmed Legitimate'],
    }
    return pd.DataFrame(change_history_data)

def main():
    # Display change history (assuming this is also part of your requirement)
    st.subheader("Change History")
    change_history_df = load_change_history_data()
    if not change_history_df.empty:
        st.dataframe(change_history_df)  # Display change history as a dataframe
    else:
        st.info("No change history available.")

if __name__ == "__main__":
    audit_logs_history_page()
    main()
