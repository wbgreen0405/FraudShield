import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

def create_anomaly_detection_plot(analyzed_df):
    """
    Creates a scatter plot for anomaly detection results.
    """
    # Assuming 'lof_scores_normalized' and 'Outlier Status' are already in 'analyzed_df'
    fig = px.scatter(
        analyzed_df, 
        x='ref_id', 
        y='lof_scores_normalized', 
        color='Outlier Status', 
        title="Anomaly Detection Scatter Plot",
        hover_data=['ref_id', 'lof_scores_normalized']
    )
    fig.update_layout(xaxis_title="Reference ID", yaxis_title="Normalized LOF Scores")
    return fig

def create_lof_distribution_plot(analyzed_df):
    """
    Creates a histogram for the distribution of LOF scores.
    """
    fig = px.histogram(
        analyzed_df,
        x='lof_scores_normalized',
        title="Distribution of Local Outlier Factor Scores",
        nbins=30  # Adjust based on your data
    )
    return fig

def app():
    st.title("Anomaly Detection System Dashboard")

    if 'analyzed_df' in st.session_state:
        analyzed_df = st.session_state['analyzed_df']

        # Check and ensure 'Outlier Status' column exists in 'analyzed_df'
        # This step assumes 'Outlier Status' should already be part of your DataFrame
        # Add or update 'Outlier Status' column here if necessary
        # Example placeholder; replace with your actual logic if 'Outlier Status' is missing
        if 'Outlier Status' not in analyzed_df.columns:
            # Placeholder logic for determining 'Outlier Status'; adjust as necessary
            analyzed_df['Outlier Status'] = ['Normal' if x <= threshold else 'Outlier' for x in analyzed_df['lof_scores_normalized']]
            # Ensure you have defined 'threshold' appropriately elsewhere in your code

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Anomaly Scatter Plot")
            scatter_fig = create_anomaly_detection_plot(analyzed_df)
            st.plotly_chart(scatter_fig)

        with col2:
            st.subheader("LOF Scores Distribution")
            dist_fig = create_lof_distribution_plot(analyzed_df)
            st.plotly_chart(dist_fig)

        st.subheader("Detailed Anomaly Transactions")
        outliers_df = analyzed_df[analyzed_df['Outlier Status'] == 'Outlier']
        st.dataframe(outliers_df, use_container_width=True)

    else:
        st.error("No analyzed data available. Please run the analysis first.")

if __name__ == '__main__':
    app()
