import streamlit as st
import pandas as pd
import plotly.express as px

def app():
    st.title("Approval System")
    
    # Assuming 'supervised_df' is already set in the session state by the main analysis function
    if 'supervised_df' in st.session_state:
        supervised_df = st.session_state['supervised_df']
        
        # Ensure 'rf_predicted_fraud' column is present
        if 'rf_predicted_fraud' in supervised_df.columns:
            # Map numeric predictions to string labels for clarity
            supervised_df['Fraud Status'] = supervised_df['rf_predicted_fraud'].map({1: 'Fraud', 0: 'Not Fraud'})
            
            # Layout with two columns: one for the DataFrame, the other for the Plotly bar chart
            col1, col2 = st.columns([2, 3])  # Adjust the ratio as needed
            
            with col1:
                st.write("Approval System Results:")
                st.dataframe(supervised_df[['ref_id', 'Fraud Status']], use_container_width=True)  # Use 'ref_id' as the transaction identifier
            
            with col2:
                # Create a Plotly bar chart for Fraud Status
                fig = px.bar(supervised_df, x='Fraud Status', color='Fraud Status', title="Fraud Status Distribution", height=400)  # Adjust height as needed
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.error("The 'rf_predicted_fraud' column is missing from the approval system data.")
    else:
        st.error('Approval system data not found.')

if __name__ == '__main__':
    st.set_page_config(page_title="Approval System", layout="wide")
    app()




