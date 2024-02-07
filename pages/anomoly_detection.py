import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

def create_anomaly_detection_plot(analyzed_df):
    # Filter for Non-Fraud transactions
    non_fraud_df = analyzed_df[analyzed_df['Approval Status'] == 'Non-Fraud']
    
    # Assign 'Outlier Status' based on LOF predictions
    non_fraud_df['Outlier Status'] = non_fraud_df['lof_predicted_fraud'].map({-1: 'Outlier', 1: 'Inlier'})

    # Create scatter plot with 'ref_id' on the x-axis and 'lof_scores_normalized' on the y-axis
    fig = px.scatter(
        non_fraud_df,
        x='ref_id',
        y='lof_scores_normalized',
        color='Outlier Status',  # Color by Outlier Status for clarity
        title="Anomaly Detection Scatter Plot",
        hover_data=['ref_id', 'lof_scores_normalized']  # Ensure 'lof_scores' is included in your dataframe
    )

    # Update layout if needed
    fig.update_layout(xaxis_title="Reference ID", yaxis_title="Normalized LOF Scores")

    return fig

def create_lof_distribution_plot(analyzed_df):
    # Filter for Non-Fraud transactions
    non_fraud_df = analyzed_df[analyzed_df['Approval Status'] == 'Non-Fraud']

    fig = px.histogram(
        non_fraud_df,
        x='lof_scores',
        nbins=20,  # Adjust the number of bins as needed
        title="Distribution of Local Outlier Factor Scores"
    )
    return fig


def app():
    st.title("Anomaly Detection System Dashboard")

    if 'analyzed_df' in st.session_state:
        analyzed_df = st.session_state['analyzed_df']

        # Normalize LOF scores for visualization
        analyzed_df['lof_scores_normalized'] = (analyzed_df['lof_scores'] - analyzed_df['lof_scores'].min()) / (analyzed_df['lof_scores'].max() - analyzed_df['lof_scores'].min())
        # Assign 'Outlier Status' based on LOF predictions
        analyzed_df['Outlier Status'] = analyzed_df['lof_predicted_fraud'].map({-1: 'Outlier', 1: 'Inlier'})

        # Plot scatter plot based on analyzed_df
        st.subheader("Anomaly Scatter Plot")
        fig = create_anomaly_detection_plot(analyzed_df)
        st.plotly_chart(fig)

        # Display detailed transactions for Outliers only
        st.subheader("Detailed Anomaly Transactions")
        # Filter for outliers
        outliers_df = analyzed_df[analyzed_df['Outlier Status'] == 'Outlier']
        # Drop unwanted columns
        desired_columns = ['ref_id', 'lof_scores'] + [col for col in analyzed_df.columns if col not in ['rf_prob_scores', 'rf_predicted_fraud', 'lof_predicted_fraud', 'Approval Status', 'Outlier Status', 'lof_scores_normalized']]
        outliers_df = outliers_df[desired_columns]
        st.dataframe(outliers_df, use_container_width=True)
    else:
        st.error("No analyzed data available. Please run the analysis first.")

if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection System Dashboard", layout="wide")
    app()


