import streamlit as st
import pandas as pd
import plotly.graph_objs as go

def create_anomaly_detection_plot(anomaly_df):
    # Normalize LOF scores for visualization
    analyzed_df['lof_scores_normalized'] = (analyzed_df['lof_scores'] -analyzed_df['lof_scores'].min()) / (analyzed_df['lof_scores'].max() - analyzed_df['lof_scores'].min())
    
    # Assign 'Outlier Status' based on LOF predictions
    analyzed_df['Outlier Status'] = analyzed_df['lof_predicted_fraud'].map({-1: 'Outlier', 1: 'Inlier'})

    # Create scatter plot
    fig = go.Figure()

    # Add inliers
    fig.add_trace(go.Scatter(
        x=anomaly_df[analyzed_df['Outlier Status'] == 'Inlier']['income'],
        y=anomaly_df[analyzed_df['Outlier Status'] == 'Inlier']['name_email_similarity'],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Inlier',
        hovertext=analyzed_df[analyzed_df['Outlier Status'] == 'Inlier']['ref_id']
    ))

    # Add outliers with size representing the LOF score
    fig.add_trace(go.Scatter(
        x=anomaly_df[analyzed_df['Outlier Status'] == 'Outlier']['income'],
        y=anomaly_df[analyzed_df['Outlier Status'] == 'Outlier']['name_email_similarity'],
        mode='markers',
        marker=dict(size=100 * analyzed_df[analyzed_df['Outlier Status'] == 'Outlier']['lof_scores_normalized'], color='red', line=dict(width=2, color='DarkSlateGrey')),
        name='Outlier',
        hovertext=anomaly_df[analyzed_df['Outlier Status'] == 'Outlier']['ref_id']
    ))

    fig.update_layout(title="Anomaly Detection Scatter Plot", xaxis_title="Income", yaxis_title="Name Email Similarity")

    return fig

def app():
    st.title("Anomaly Detection System Dashboard")

    if 'analyzed_df' in st.session_state:
        analyzed_df = st.session_state['analyzed_df']

        # Plot scatter plot
        st.subheader("Anomaly Scatter Plot")
        fig = create_anomaly_detection_plot(analyzed_df)
        st.plotly_chart(fig)

        # Display detailed transactions
        st.subheader("Detailed Anomaly Transactions")
        st.dataframe(analyzed_df[['ref_id', 'income', 'name_email_similarity', 'lof_scores']], use_container_width=True)
    else:
        st.error("No anomalies found. Run the analysis to detect anomalies.")

if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection System Dashboard", layout="wide")
    app()



