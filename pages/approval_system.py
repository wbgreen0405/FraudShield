import streamlit as st
import pandas as pd
import plotly.express as px

def app():
    st.title("Approval System Dashboard")

    if 'supervised_df' in st.session_state:
        supervised_df = st.session_state['supervised_df']
        supervised_df['Fraud Status'] = supervised_df['rf_predicted_fraud'].map({1: 'Fraud', 0: 'Not Fraud'})

        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("Transactions Overview")
            fraud_count = supervised_df['Fraud Status'].value_counts().get('Fraud', 0)
            non_fraud_count = supervised_df['Fraud Status'].value_counts().get('Not Fraud', 0)
            st.metric("Fraud Transactions", fraud_count)
            st.metric("Non-Fraud Transactions", non_fraud_count)

        with col2:
            st.subheader("Fraud Status Distribution")
            fig = px.bar(supervised_df, x='Fraud Status', color='Fraud Status', title="Fraud Status Distribution")
            st.plotly_chart(fig)

        st.subheader("Detailed Transactions")
        status_filter = st.selectbox("Select Fraud Status to View", ["All", "Fraud", "Not Fraud"])

        if status_filter != "All":
            filtered_df = supervised_df[supervised_df['Fraud Status'] == status_filter]
        else:
            filtered_df = supervised_df

        st.dataframe(filtered_df[['ref_id', 'Fraud Status']], use_container_width=True)

        if 'anomaly_df' in st.session_state:
            with st.expander("Explore Anomalies Detected"):
                st.dataframe(st.session_state['anomaly_df'])

        if 'review_df' in st.session_state:
            with st.expander("Offline Review Detailed Transactions"):
                st.dataframe(st.session_state['review_df'])

    else:
        st.error("No transaction data available. Please run the analysis first.")

if __name__ == '__main__':
    st.set_page_config(page_title="Approval System Dashboard", layout="wide")
    app()




