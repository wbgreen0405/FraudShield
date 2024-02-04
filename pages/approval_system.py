import streamlit as st
import pandas as pd
import plotly.express as px

def app():
    st.title("Approval System")
    
    if 'supervised_df' in st.session_state:
        supervised_df = st.session_state['supervised_df']
        
        if 'rf_predicted_fraud' in supervised_df.columns:
            supervised_df['Fraud Status'] = supervised_df['rf_predicted_fraud'].map({1: 'Fraud', 0: 'Not Fraud'})
            
            # Adjust layout: two columns for DataFrame and Plotly bar chart
            col1, col2 = st.columns([2, 2])  # Making them equal for alignment
            
            with col1:
                st.write("Approval System Results:")
                # Use 'ref_id' as the transaction identifier and drop the index column
                st.dataframe(supervised_df[['ref_id', 'Fraud Status']].reset_index(drop=True), height=600)  # Adjust height as needed
            
            with col2:
                # Create a Plotly bar chart for Fraud Status with adjusted height for alignment
                fig = px.bar(supervised_df, x='Fraud Status', color='Fraud Status', title="Fraud Status Distribution")
                fig.update_layout(autosize=False, height=600)  # Match the height with the DataFrame
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.error("The 'rf_predicted_fraud' column is missing from the approval system data.")
    else:
        st.error('Approval system data not found.')

if __name__ == '__main__':
    st.set_page_config(page_title="Approval System", layout="wide")
    app()




