import streamlit as st
from streamlit_option_menu import option_menu
# Import page functions from the pages package
from src import (
    home_app as home,
    transactions_app as transactions,
    approval_system_app as approval_system,
    anomaly_detection_app as anomaly_detection,
    case_detail_app as case_detail,
    test_and_learn_loop_app as test_and_learn_loop,
    help_documentation_app as help_documentation,
    audit_logs_history_app as audit_logs_history
)



st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")


def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu", 
            ["Home", "Transaction Analysis", "Approval System", "Anomaly Detection", "Case Detail", "Test and Learn Loop", "Help / Documentation", "Audit Logs / History"],
            icons=["house", "credit-card", "check2-circle", "eye", "file-earmark-text", "bi bi-arrow-clockwise", "bar-chart-line", "book", "file-earmark-text"],
            menu_icon="cast", 
            default_index=0
        )




    # Automatically start Transaction Analysis when the app is first initiated or when selected from the menu
    if 'analysis_performed' not in st.session_state or selected == "Transaction Analysis":
        transactions_app.app()  # Run Transaction Analysis
    else:
        # Define pages in a dictionary
        pages = {
            "Home": home,
            "Approval System": approval_system,
            "Anomaly Detection": anomaly_detection,
            "Case Detail": case_detail,
            "Test and Learn Loop": test_and_learn_loop,
            "Help / Documentation": help_documentation,
            "Audit Logs / History": audit_logs_history
        }
        
        # Call the appropriate page function based on the user's selection
        if selected in pages:
            pages[selected]()

# Call the main function when the script is run
# Make sure to call the main function
if __name__ == "__main__":
    main()
