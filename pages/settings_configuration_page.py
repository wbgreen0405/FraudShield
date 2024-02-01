import streamlit as st

def settings_configuration_page():
    st.title("Settings / Configuration")

    # Configure the fraud detection threshold
    fraud_threshold = st.slider("Fraud Detection Threshold", min_value=0.0, max_value=1.0, value=0.5, key='fraud_threshold')
    
    st.success("Configuration updated.")

    st.markdown("### More Settings")
    # Add more settings here as needed
    
    if st.button("Apply Changes"):
        # Update session_state (though it's automatically updated with user input)
        st.session_state['fraud_threshold'] = fraud_threshold
        st.success("Changes applied successfully!")
