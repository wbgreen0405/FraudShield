# Home page script
import streamlit as st

def home_page():
    st.title('Welcome to FraudShield: The Fraud Detection System Demo.')

    # Placeholder for the pipeline image
    st.subheader("Fraud Detection Pipeline")
    st.image("assets/logo_transparent.png", caption="Fraud Detection Pipeline Visualization", use_column_width=True)
    
    # Placeholder for high-level statistics
    st.subheader("High-Level Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", "0")  # Replace "0" with dynamic count after data upload
    with col2:
        st.metric("Transactions Flagged", "0")  # Replace "0" with dynamic count after data upload
    with col3:
        st.metric("Transactions Approved", "0")  # Replace "0" with dynamic count after data upload
    with col4:
        st.metric("Transactions Marked as Fraudulent", "0")  # Replace "0" with dynamic count after data upload

    # Rest of the home page content
    st.markdown("""
    Dive into the world of anti-fraud analytics with our streamlined, user-friendly platform designed to highlight suspicious financial activities through data-driven insights. FraudShield provides a comprehensive suite of tools to identify, analyze, and combat fraudulent transactions, ensuring your financial security.
    
    **Key Features:**
    - Data Upload and Preprocessing: Securely upload and preprocess your transactional data.
    - Model Predictions: Use a pre-trained Random Forest classifier to identify potential fraud cases.
    - Anomaly Detection: Apply the Local Outlier Factor model to scrutinize flagged transactions.
    - Human Judgment Simulation: Simulate expert review decisions to validate flagged cases.
    - Review and Update: Update the dataset based on the review outcomes and prepare it for further analysis or model retraining.
    
    **How to Use:** Navigate through the sidebar to access different stages of the fraud detection process. Start by uploading your dataset and then proceed through each stage to explore the full capabilities of the system.
    
    Inspired by innovative methodologies in fraud detection, this app offers an in-depth look at how data science can be leveraged to protect financial assets and maintain transactional integrity.
    """)

# Call the home_page function if this script is run
if __name__ == '__main__':
    home_page()
