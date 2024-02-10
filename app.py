import streamlit as st
# Import pages for each section of your app
from pages import (
    transactions,
    approval_system,
    anomaly_detection,
    case_detail,
    test_and_learn_loop_page,
    help_documentation_page,
    audit_logs_history_page,
    #supervised_fraud_results_page  # Import the new page here
)

# Define a simple navigation structure using Streamlit sidebar
st.sidebar.title('Navigation')
# Updated to include new pages
page = st.sidebar.selectbox(
    "Choose a page", [
        "Home",
        "Transaction Analysis",
        "Approval System",
        "Anomaly Detection",
        "Case Detail",
        "Test and Learn Loop",
        "Help / Documentation",
        "Audit Logs / History",
        #"Supervised Fraud Results"  # Add the new page to the dropdown
    ]
)

# Initialize session state for navigation if it doesn't exist
if 'navigate_to_approval' not in st.session_state:
    st.session_state['navigate_to_approval'] = False

# Page rendering based on navigation state
if page == "Home":
    st.write("Welcome to the Home page. Select an option from the sidebar to get started.")
elif page == "Transaction Analysis":
    transactions.app()
elif page == "Approval System":
    approval_system.app()
elif page == "Anomaly Detection":
    anomaly_detection.app()
elif page == "Case Detail":
    case_detail.app()
elif page == "Test and Learn Loop":
    test_and_learn_loop_page.app()
elif page == "Help / Documentation":
    help_documentation_page.app()
elif page == "Audit Logs / History":
    audit_logs_history_page.app()
#elif page == "Supervised Fraud Results":  # Add the rendering condition for the new page
    #supervised_fraud_results_page.app()  # Make sure this matches the function name in your new page module






