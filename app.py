import os
import streamlit as st
from pages.home import home_page
from pages.transactions import transactions_page
from pages.offline_review import offline_review_page
from pages.supervised_fraud_results import supervised_fraud_results_page
from pages.fraud_rules_page import fraud_rules_page
from pages.anomaly_detection_system_page import anomaly_detection_system_page
from pages.test_and_learn_page import test_and_learn_page
from pages.settings_configuration_page import settings_configuration_page
from pages.audit_logs_history_page import audit_logs_history_page
from pages.model_performance_metrics_page import model_performance_metrics_page
from pages.help_documentation_page import help_documentation_page
from pages.expert_human_judgment_page import expert_human_judgment_page


def render_sidebar():
    # Your existing sidebar rendering code
    pages = {
        'Home': home_page,
        'Transactions': transactions_page,
        'Supervised Fraud Results': supervised_fraud_results_page,
        'Offline Review': offline_review_page,
        'Fraud Rules': fraud_rules_page,
        'Anomaly Detection': anomaly_detection_system_page,
        'Expert Human Judgment': expert_human_judgment_page,
        'Test and Learn': test_and_learn_page,
        'Settings / Configuration': settings_configuration_page,
        'Audit Logs / History': audit_logs_history_page,
        'Model Performance Metrics': model_performance_metrics_page,
        'Help / Documentation': help_documentation_page,
    }

    # Your existing code to display the sidebar and handle page navigation
    page = st.sidebar.radio("Select a page:", list(pages.keys()))
    page_function = pages[page]
    page_function()

def main():
    render_sidebar()

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()

