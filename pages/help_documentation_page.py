import streamlit as st

def help_documentation_page():
    st.title("Help / Documentation")
    
    st.write("""
    ## Understanding the Fraud Detection Workflow

    This page serves as a guide to understanding the fraud detection system implemented in our platform. 
    Below is a step-by-step walkthrough of the process, supplemented by the diagram provided by Aditya Khandekar.

    ### Fraud Detection Model

    - **Supervised Fraud Model**: Our system uses predictive modeling and complex pattern analysis to detect fraudulent activities. This model is trained on historical data to recognize fraudulent transactions.
    - **Fraud Rules**: We apply a set of rules, both internal and external, to assist the model in decision-making.

    ### Approval System

    - **Marked as Fraud**: If the supervised fraud model flags a transaction as fraudulent, it moves to the anomaly detection system for further analysis.
    - **Marked as Approved**: Transactions not flagged by the fraud model are approved.

    ### Anomaly Detection System

    - **Suspected Fraud**: The anomaly detection system analyzes flagged transactions to determine if they are outliers based on the user's context and historical data.

    ### Expert Human Judgment

    - **Offline Process**: Transactions deemed suspicious by the anomaly detection system are reviewed offline by experts. They confirm whether these transactions are fraudulent or legitimate.

    ### Continuous Learning

    - **Test and Learn Loop**: The outcomes of the expert review feed back into the system, improving both the supervised fraud model and the anomaly detection system, reducing false positives and negatives over time.

    ### Contact for Support

    For more detailed support, please reach out to our team at [support@example.com](mailto:support@example.com).

    ### Diagram

    ![Fraud Detection Workflow](path_to_diagram_image)
    """)

    # Optionally, if you want to display the diagram within Streamlit from an uploaded file
    #st.image("path_to_diagram_image", caption="Fraud Detection Workflow Diagram")

help_documentation_page()

