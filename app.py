import os
import streamlit as st
import datetime

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

# Import other necessary functions and pages
from pages import home
from pages import transactions
from pages import supervised_fraud_results_page
from pages import fraud_rules_page
from pages import anomaly_detection_system_page
from pages import test_and_learn_page
from pages import settings_configuration_page
from pages import audit_logs_history_page
from pages import model_performance_metrics_page
from pages import help_documentation_page
from pages import expert_human_judgment_page


def render_sidebar():
    # Your existing sidebar rendering code
    pages = {
        'Home': home.home_page,
        'Transactions': transactions.transactions_page,
        'Supervised Fraud Results': supervised_fraud_results_page.supervised_fraud_results_page,
        'Fraud Rules': fraud_rules_page.fraud_rules_page,
        'Anomaly Detection': anomaly_detection_system_page.anomaly_detection_system_page,
        'Expert Human Judgment': expert_human_judgment_page.expert_human_judgment_page,
        'Test and Learn': test_and_learn_page.test_and_learn_page,
        'Settings / Configuration': settings_configuration_page.settings_configuration_page,
        'Audit Logs / History': audit_logs_history_page.audit_logs_history_page,
        'Model Performance Metrics': model_performance_metrics_page.model_performance_metrics_page,
        'Help / Documentation': help_documentation_page.help_documentation_page,
    }

    # Your existing code to display the sidebar and handle page navigation
    page = st.sidebar.radio("Select a page:", list(pages.keys()))
    page_function = pages[page]
    page_function()

def main():
    render_sidebar()

if __name__ == '__main__':
    main()




