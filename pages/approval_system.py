import streamlit as st
import plotly.express as px

def app():
    # Corrected to match the DataFrame name used in the main analysis
    if 'approval_system_df' in st.session_state:
        approval_df = st.session_state['approval_system_df']
        
        # Display the DataFrame with an emphasis on 'Approval Status'
        st.write("Approval System Data:")
        st.dataframe(approval_df[['Transaction ID', 'Approval Status', 'Other Relevant Columns']])  # Ensure you replace 'Transaction ID' and 'Other Relevant Columns' with actual column names from your DataFrame
        
        # Generate a bar chart showing counts of 'Fraud' vs. 'Non-Fraud' based on 'Approval Status'
        if 'Approval Status' in approval_df.columns:
            status_counts = approval_df['Approval Status'].value_counts().reset_index()
            status_counts.columns = ['Approval Status', 'Count']
            fig = px.bar(status_counts, x='Approval Status', y='Count', title="Fraud vs. Non-Fraud Transactions",
                         color='Approval Status', text='Count')
            st.plotly_chart(fig)
        else:
            st.error("The 'Approval Status' column is missing from the approval data.")
    else:
        st.error('Approval system data not found.')

# This line is optional, used for direct script testing
if __name__ == '__main__':
    st.set_page_config(page_title="Approval System", layout="wide")
    app()


