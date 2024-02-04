import streamlit as st
import plotly.express as px

def app():
    st.title("Approval System")
    
    # Assuming approval_df is accessible; you might need to adjust how it's passed between pages
    approval_df = get_approval_df_somehow()

    # Display the DataFrame
    st.write(approval_df)
    
    # Generate and display a bar chart of fraud and non-fraud transactions
    fig = px.bar(approval_df, x='Approval Status', title="Fraud vs. Non-Fraud Transactions")
    st.plotly_chart(fig)
    
if __name__ == '__main__':
    st.set_page_config(page_title="Approval System", layout="wide")
    app()
