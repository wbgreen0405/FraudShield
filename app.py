import streamlit as st
from streamlit_option_menu import option_menu
# Import page functions from the pages package
from pages import (
    home_app as home,
    transactions_app as transactions,
    approval_system_app as approval_system,
    anomaly_detection_app as anomaly_detection,
    case_detail_app as case_detail,
    test_and_learn_loop_app as test_and_learn_loop,
    help_documentation_app as help_documentation,
    audit_logs_history_app as audit_logs_history
)

st.set_page_config(page_title="Home", layout="wide")

def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu", 
            ["Home", "Transaction Analysis", "Approval System", "Anomaly Detection", "Case Detail", "Test and Learn Loop", "Supervised Fraud Results", "Help / Documentation", "Audit Logs / History"],
            icons=["house", "credit-card", "check2-circle", "eye", "file-earmark-text", "bi bi-arrow-clockwise", "bar-chart-line", "book", "file-earmark-text"],
            menu_icon="cast", 
            default_index=0
        )

    # Call the appropriate page function based on the user's selection
    if selected == "Home":
        home()
    elif selected == "Transaction Analysis":
        transactions()
    elif selected == "Approval System":
        approval_system()
    elif selected == "Anomaly Detection":
        anomaly_detection()
    elif selected == "Case Detail":
        case_detail()
    elif selected == "Test and Learn Loop":
        test_and_learn_loop()
    elif selected == "Supervised Fraud Results":
        supervised_fraud_results()
    elif selected == "Help / Documentation":
        help_documentation()
    elif selected == "Audit Logs / History":
        audit_logs_history()

# Call the main function when the script is run
if __name__ == '__main__':
   
    main()



