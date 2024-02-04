import streamlit as st
import plotly.express as px

def app():
    if 'approval_df' in st.session_state:
        approval_df = st.session_state['approval_df']
        
        # Display the DataFrame
        st.write("Approval Data:", approval_df)
        
        # Check if 'Approval Status' column exists and generate a bar chart
        if 'Approval Status' in approval_df.columns:
            fig = px.bar(approval_df, x='Approval Status', title="Fraud vs. Non-Fraud Transactions",
                         color='Approval Status', barmode='group')
            st.plotly_chart(fig)
        else:
            st.error("The 'Approval Status' column is missing from the approval data.")
    else:
        st.error('Approval data not found.')

# This line is optional, used for direct script testing
if __name__ == '__main__':
    st.set_page_config(page_title="Approval System", layout="wide")
    app()

