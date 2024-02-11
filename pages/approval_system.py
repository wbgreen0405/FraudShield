import streamlit as st
import pandas as pd
import plotly.express as px

def app():
    st.title("Approval System Page")

    if 'supervised_df' in st.session_state:
        supervised_df = st.session_state['supervised_df']
        
        # Map 'rf_predicted_fraud' to a more readable format and rename the column
        supervised_df['Flagged as Fraud'] = supervised_df['rf_predicted_fraud'].map({1: 'Yes', 0: 'No'})
        
        # Drop 'lof_predicted_fraud' and 'Approval Status' if present
        columns_to_drop = ['lof_predicted_fraud', 'Approval Status','lof_scores', 'lof_scores_normalized']
        for col in columns_to_drop:
            if col in supervised_df.columns:
                supervised_df = supervised_df.drop(columns=[col])


        # Rearrange columns to have 'ref_id' and 'Flagged as Fraud' first
        cols = ['ref_id', 'Flagged as Fraud','rf_prob_scores'] + [col for col in supervised_df.columns if col not in ['ref_id', 'Flagged as Fraud','rf_prob_scores']]
        supervised_df = supervised_df[cols]

        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("Transactions Overview")
            fraud_count = supervised_df['Flagged as Fraud'].value_counts().get('Yes', 0)
            non_fraud_count = supervised_df['Flagged as Fraud'].value_counts().get('No', 0)
            st.metric("Fraud Transactions", fraud_count)
            st.metric("Non-Fraud Transactions", non_fraud_count)

        with col2:
            st.subheader("Fraud Status Distribution")
            fig = px.bar(supervised_df, x='Flagged as Fraud', color='Flagged as Fraud', title="Fraud Status Distribution")
            st.plotly_chart(fig)

        st.subheader("Detailed Transactions")
        status_filter = st.selectbox("Select Fraud Status to View", ["All", "Yes", "No"])

        if status_filter != "All":
            filtered_df = supervised_df[supervised_df['Flagged as Fraud'] == status_filter]
        else:
            filtered_df = supervised_df

        st.dataframe(filtered_df, use_container_width=True)

    else:
        st.error("No transaction data available. Please run the analysis first.")

#if __name__ == '__main__':
    #st.set_page_config(page_title="Approval System Dashboard", layout="wide")
app()




