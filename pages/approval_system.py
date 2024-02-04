import streamlit as st
import pandas as pd
#from pygwalker.streamlit import st_pygwalker  # Ensure Pygwalker is installed and this import works
import pygwalker as pyg

def app():
    # Assuming 'supervised_df' is already set in the session state by the main analysis function
    if 'supervised_df' in st.session_state:
        supervised_df = st.session_state['supervised_df']
        
        # Ensure 'rf_predicted_fraud' column is present
        if 'rf_predicted_fraud' in supervised_df.columns:
            # Map numeric predictions to string labels for clarity
            supervised_df['Fraud Status'] = supervised_df['rf_predicted_fraud'].map({1: 'Fraud', 0: 'Not Fraud'})
            
            # Display the DataFrame with an emphasis on the Fraud Status
            st.write("Approval System Results:")
            st.dataframe(supervised_df[['ref_id', 'Fraud Status']], use_container_width=True)  # Use 'ref_id' as the transaction identifier
            
            # Create a Pygwalker bar chart for Fraud Status
            fraud_status_counts = supervised_df['Fraud Status'].value_counts().reset_index()
            fraud_status_counts.columns = ['Fraud Status', 'Count']
            pyg.st_pygwalker(fraud_status_counts, kind='bar', x='Fraud Status', y='Count', title="Fraud Status Distribution")
            
        else:
            st.error("The 'rf_predicted_fraud' column is missing from the approval system data.")
    else:
        st.error('Approval system data not found.')

if __name__ == '__main__':
    st.set_page_config(page_title="Approval System", layout="wide")
    app()




