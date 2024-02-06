import streamlit as st
import pandas as pd
import plotly.graph_objs as go

def create_anomaly_detection_plot(anomaly_df):
    # Normalize LOF scores for visualization
    anomaly_df['lof_scores_normalized'] = (anomaly_df['lof_scores'] - anomaly_df['lof_scores'].min()) / (anomaly_df['lof_scores'].max() - anomaly_df['lof_scores'].min())
    
    # Assign 'Outlier Status' based on LOF predictions
    anomaly_df['Outlier Status'] = anomaly_df['lof_predicted_fraud'].map({-1: 'Outlier', 1: 'Inlier'})

    # Create scatter plot
    fig = go.Figure()

    # Add inliers
    fig.add_trace(go.Scatter(
        x=anomaly_df[anomaly_df['Outlier Status'] == 'Inlier']['income'],
        y=anomaly_df[anomaly_df['Outlier Status'] == 'Inlier']['name_email_similarity'],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Inlier',
        hovertext=anomaly_df[anomaly_df['Outlier Status'] == 'Inlier']['ref_id']
    ))

    # Add outliers with size representing the LOF score
    fig.add_trace(go.Scatter(
        x=anomaly_df[anomaly_df['Outlier Status'] == 'Outlier']['income'],
        y=anomaly_df[anomaly_df['Outlier Status'] == 'Outlier']['name_email_similarity'],
        mode='markers',
        marker=dict(size=100 * anomaly_df[anomaly_df['Outlier Status'] == 'Outlier']['lof_scores_normalized'], color='red', line=dict(width=2, color='DarkSlateGrey')),
        name='Outlier',
        hovertext=anomaly_df[anomaly_df['Outlier Status'] == 'Outlier']['ref_id']
    ))

    fig.update_layout(title="Anomaly Detection Scatter Plot", xaxis_title="Income", yaxis_title="Name Email Similarity")

    return fig

def app():
    st.title("Anomaly Detection System Dashboard")

    if 'anomaly_df' in st.session_state:
        anomaly_df = st.session_state['anomaly_df']

        # Plot scatter plot
        st.subheader("Anomaly Scatter Plot")
        fig = create_anomaly_detection_plot(anomaly_df)
        st.plotly_chart(fig)

        # Display detailed transactions
        st.subheader("Detailed Anomaly Transactions")
        st.dataframe(anomaly_df[['ref_id', 'income', 'name_email_similarity', 'lof_scores']], use_container_width=True)
    else:
        st.error("No anomalies found. Run the analysis to detect anomalies.")

if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection System Dashboard", layout="wide")
    app()
