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
    help_documentation_app as help_documentation
)

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu", 
            ["Home", "Transaction Analysis", "Approval System", "Anomaly Detection", "Case Detail", "Test and Learn Loop", "Help / Documentation"],
            icons=["house", "credit-card", "check2-circle", "eye", "file-earmark-text", "bi bi-arrow-clockwise", "bar-chart-line", "book"],
            menu_icon="cast", 
            default_index=0
        )

    # Define pages in a dictionary
    pages = {
        "Home": home,
        "Transaction Analysis": transactions,
        "Approval System": approval_system,
        "Anomaly Detection": anomaly_detection,
        "Case Detail": case_detail,
        "Test and Learn Loop": test_and_learn_loop,
        "Help / Documentation": help_documentation,
      
    }
    
    # Call the appropriate page function based on the user's selection with a spinner
    if selected in pages:
        with st.spinner(f"Loading {selected}..."):
            pages[selected]()

if __name__ == "__main__":
    main()
