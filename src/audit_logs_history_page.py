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
    return pd.DataFrame(audit_logs)

def load_change_history_data():
    change_history_data = {
        'Timestamp': [datetime.now(), datetime.now()],
        'User': ['User A', 'User B'],
        'Change': ['Decision Updated', 'Decision Confirmed'],
        'Details': ['Decision: Confirmed Fraud', 'Decision: Confirmed Legitimate'],
    }
    return pd.DataFrame(change_history_data)

def app():
    st.title("Audit Logs / History")
    st.write("View detailed transaction audit logs and the history of changes made through the UI.")
    
    if 'audit_logs' not in st.session_state:
        st.session_state['audit_logs'] = []
    
    if 'outcome_df' in st.session_state and st.session_state['outcome_df'] is not None:
        outcome_dff = st.session_state['outcome_df']
        for _, row in outcome_df.iterrows():
            log_audit_entry(
                transaction_id=row['ref_id'],
                reviewer_id="Expert Reviewer",
                decision=row.get('expert_decision', 'No Decision'),
                audit_logs=st.session_state['audit_logs']
            )
    
    audit_logs_df = load_audit_logs_data(st.session_state['audit_logs'])
    st.subheader("Transaction Audit Logs")
    if not audit_logs_df.empty:
        st.dataframe(audit_logs_df)
    else:
        st.info("No audit logs available.")
    
    st.subheader("Change History")
    change_history_df = load_change_history_data()
    if not change_history_df.empty:
        st.dataframe(change_history_df)
    else:
        st.info("No change history available.")


app()
