import streamlit as st
from streamlit_option_menu import option_menu
from pages import (
    home_app,
    transactions_app,
    approval_system_app,
    anomaly_detection_app,
    case_detail_app,
    test_and_learn_loop_page_app,
    help_documentation_page_app,
    audit_logs_history_page_app,
    supervised_fraud_results_page_app,
)

def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu", 
            ["Home", "Transaction Analysis", "Approval System", "Anomaly Detection", "Case Detail", "Test and Learn Loop", "Supervised Fraud Results", "Help / Documentation", "Audit Logs / History"],
            icons=["house", "credit-card", "check2-circle", "eye", "file-earmark-text", "bi bi-arrow-clockwise", "bar-chart-line", "book", "file-earmark-text"],
            menu_icon="cast", 
            default_index=0
        )

    # Page rendering based on the navigation choice
    if selected == "Home":
        home_app()
    elif selected == "Transaction Analysis":
        transactions_app()
    elif selected == "Approval System":
        approval_system_app()
    elif selected == "Anomaly Detection":
        anomaly_detection_app()
    elif selected == "Case Detail":
        case_detail_app()
    elif selected == "Test and Learn Loop":
        test_and_learn_loop_page_app()
    elif selected == "Supervised Fraud Results":
        supervised_fraud_results_page_app()
    elif selected == "Help / Documentation":
        help_documentation_page_app()
    elif selected == "Audit Logs / History":
        audit_logs_history_page_app()

if __name__ == '__main__':
    main()



