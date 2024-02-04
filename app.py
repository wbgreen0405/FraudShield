import streamlit as st
from pages import transactions, approval_system  # Make sure this import path matches your directory structure

# Define a simple navigation structure using Streamlit sidebar
st.sidebar.title('Navigation')
page = st.sidebar.selectbox("Choose a page", ["Transaction Analysis", "Approval System"])

# Initialize session state for navigation if it doesn't exist
if 'navigate_to_approval' not in st.session_state:
    st.session_state['navigate_to_approval'] = False

# Page rendering based on navigation state
if page == "Transaction Analysis":
    # Check if we should navigate to the Approval System page
    if st.session_state['navigate_to_approval']:
        # Reset the state and redirect to the Approval System page
        st.session_state['navigate_to_approval'] = False
        approval_system.app()
    else:
        transactions.app()
elif page == "Approval System":
    approval_system.app()

# Optionally, add additional pages as needed




