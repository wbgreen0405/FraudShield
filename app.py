import streamlit as st
# Import pages for each section of your app
from pages import transactions, approval_system, anomaly_detection, case_detail

# Define a simple navigation structure using Streamlit sidebar
st.sidebar.title('Navigation')
# Updated to include new pages
page = st.sidebar.selectbox("Choose a page", ["Home", "Transaction Analysis", "Approval System", "Anomaly Detection", "Case Detail"])

# Initialize session state for navigation if it doesn't exist
if 'navigate_to_approval' not in st.session_state:
    st.session_state['navigate_to_approval'] = False

# Page rendering based on navigation state
if page == "Home":
    st.write("Welcome to the Home page. Select an option from the sidebar to get started.")
elif page == "Transaction Analysis":
    # Check if we should navigate to the Approval System page
    if st.session_state['navigate_to_approval']:
        # Reset the state and redirect to the Approval System page
        st.session_state['navigate_to_approval'] = False
        approval_system.app()
    else:
        transactions.app()
elif page == "Approval System":
    approval_system.app()
elif page == "Anomaly Detection":
    anomaly_detection.app()
elif page == "Case Detail":
    case_detail.app()




