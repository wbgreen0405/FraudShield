import streamlit as st
from streamlit_option_menu import option_menu
from pages import home, transactions, approval_system, anomaly_detection, case_detail, test_and_learn_loop_page, help_documentation_page, audit_logs_history_page
# Define the navigation menu
def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu", 
            ["Home", "Transaction Analysis", "Approval System", "Anomaly Detection", "Case Detail", "Test and Learn Loop", "Supervised Fraud Results", "Help / Documentation", "Audit Logs / History"],
            icons=["house", "credit-card", "check2-circle", "eye", "file-earmark-text", "loop", "bar-chart-line", "book", "file-earmark-text"],
            menu_icon="cast", 
            default_index=0
        )

    # Page rendering based on the navigation choice
    if selected == "Home":
        home.app()
    elif selected == "Transaction Analysis":
        transactions.app()
    elif selected == "Approval System":
        approval_system.app()
    elif selected == "Anomaly Detection":
        anomaly_detection.app()
    elif selected == "Case Detail":
        case_detail.app()
    elif selected == "Test and Learn Loop":
        test_and_learn_loop_page.app()
    elif selected == "Supervised Fraud Results":
        supervised_fraud_results_page.app()
    elif selected == "Help / Documentation":
        help_documentation_page.app()
    elif selected == "Audit Logs / History":
        audit_logs_history_page.app()

if __name__ == '__main__':
    main()


