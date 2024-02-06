import streamlit as st
import pandas as pd
import plotly.express as px

def create_anomaly_detection_plot(analyzed_df):
    # Normalize LOF scores for visualization
    analyzed_df['lof_scores_normalized'] = (analyzed_df['lof_scores'] - analyzed_df['lof_scores'].min()) / (analyzed_df['lof_scores'].max() - analyzed_df['lof_scores'].min())
    
    # Assign 'Outlier Status' based on LOF predictions
    analyzed_df['Outlier Status'] = analyzed_df['lof_predicted_fraud'].map({-1: 'Outlier', 1: 'Inlier'})

    # Create scatter plot with 'ref_id' on the x-axis and 'lof_scores_normalized' on the y-axis
    fig = px.scatter(
        analyzed_df, 
        x='ref_id', 
        y='lof_scores_normalized', 
        color='Outlier Status',  # Color by Outlier Status for clarity
        title="Anomaly Detection Scatter Plot",
        hover_data=['ref_id', 'lof_scores_normalized']  # Ensure 'lof_scores' is included in your dataframe
    )
    
    # Update layout if needed
    fig.update_layout(xaxis_title="Reference ID", yaxis_title="Normalized LOF Scores")

    return fig

def app():
    st.title("Anomaly Detection System Dashboard")

    if 'analyzed_df' in st.session_state:
        analyzed_df = st.session_state['analyzed_df']

        # Plot scatter plot based on analyzed_df
        st.subheader("Anomaly Scatter Plot")
        fig = create_anomaly_detection_plot(analyzed_df)
        st.plotly_chart(fig)

        # Display detailed transactions
        st.subheader("Detailed Anomaly Transactions")
        st.dataframe(analyzed_df[['ref_id', 'lof_scores_normalized', 'Outlier Status']], use_container_width=True)
    else:
        st.error("No analyzed data available. Please run the analysis first.")

if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection System Dashboard", layout="wide")
    app()
