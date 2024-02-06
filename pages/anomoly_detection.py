import streamlit as st
import pandas as pd
import plotly.express as px

def create_anomaly_detection_plot(analyzed_df):
    # Filter for Non-Fraud transactions
    non_fraud_df = analyzed_df[analyzed_df['Approval Status'] == 'Non-Fraud']
    
    # Normalize LOF scores for visualization
    non_fraud_df['lof_scores_normalized'] = (non_fraud_df['lof_scores'] - non_fraud_df['lof_scores'].min()) / (non_fraud_df['lof_scores'].max() - non_fraud_df['lof_scores'].min())
    
    # Assign 'Outlier Status' based on LOF predictions
    non_fraud_df['Outlier Status'] = non_fraud_df['lof_predicted_fraud'].map({-1: 'Outlier', 1: 'Inlier'})

    # Create scatter plot with 'ref_id' on the x-axis and 'lof_scores' on the y-axis
    fig = px.scatter(
        non_fraud_df, 
        x='income', 
        y='lof_scores_normalized', 
        color='Outlier Status',  # Color by Outlier Status for clarity
        title="Anomaly Detection Scatter Plot",
        hover_data=['ref_id', 'lof_scores']  # Ensure 'lof_scores' is included in your dataframe
    )
    
    # Update layout if needed
    fig.update_layout(xaxis_title="Reference ID", yaxis_title="LOF Scores Normalized")

    return fig

def app():
    st.title("Anomaly Detection System Dashboard")

    if 'analyzed_df' in st.session_state:
        analyzed_df = st.session_state['analyzed_df']

        # Plot scatter plot based on Non-Fraud transactions
        st.subheader("Anomaly Scatter Plot for Non-Fraud Transactions")
        fig = create_anomaly_detection_plot(analyzed_df)
        st.plotly_chart(fig)

        # Display detailed transactions for Non-Fraud
        st.subheader("Detailed Non-Fraud Transactions")
        non_fraud_df = analyzed_df[analyzed_df['Approval Status'] == 'Non-Fraud']
        st.dataframe(non_fraud_df[['ref_id', 'lof_scores']], use_container_width=True)
    else:
        st.error("No analyzed data available. Please run the analysis first.")

if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection System Dashboard", layout="wide")
    app()

