import streamlit as st
import pandas as pd
import plotly.express as px

# Assuming lof_scores are already included and correctly assigned in your DataFrame

# Function to create anomaly detection scatter plot
def create_anomaly_detection_plot(df):
    fig = px.scatter(
        df,
        x='ref_id',
        y='lof_scores',  # Use raw LOF scores for plotting
        color='LOF Status',  # Use 'LOF Status' for coloring points
        title="Anomaly Detection Scatter Plot",
        hover_data=['ref_id', 'lof_scores']
    )
    fig.update_layout(xaxis_title="Reference ID", yaxis_title="LOF Scores")
    return fig

# Function to create LOF scores distribution plot
def create_lof_distribution_plot(df):
    fig = px.histogram(
        df,
        x='lof_scores',  # Use raw LOF scores for the histogram
        nbins=20,
        title="Distribution of Local Outlier Factor Scores"
    )
    return fig

# Streamlit app function
def app():
    st.title("Anomaly Detection System Dashboard")

    if 'analyzed_df' in st.session_state:
        analyzed_df = st.session_state['analyzed_df']

        # Setup layout for plots
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Anomaly Scatter Plot")
            scatter_fig = create_anomaly_detection_plot(analyzed_df)
            st.plotly_chart(scatter_fig)

        with col2:
            st.subheader("LOF Scores Distribution")
            dist_fig = create_lof_distribution_plot(analyzed_df)
            st.plotly_chart(dist_fig)

        # Filter for outliers based on 'LOF Status'
        outliers_df = analyzed_df[analyzed_df['LOF Status'] == 'Suspected Fraud']
        
        # Drop unwanted columns for clarity in the detailed view
        columns_to_drop = ['rf_prob_scores', 'rf_predicted_fraud', 'RF Approval Status']
        outliers_df = outliers_df.drop(columns=[col for col in columns_to_drop if col in outliers_df.columns], errors='ignore')

        # Reorder columns for the detailed view
        cols = ['ref_id', 'LOF Status', 'lof_scores'] + [col for col in outliers_df.columns if col not in ['ref_id', 'LOF Status', 'lof_scores']]
        outliers_df = outliers_df[cols]

        st.subheader("Detailed Anomaly Transactions")
        st.dataframe(outliers_df, use_container_width=True)
    else:
        st.error("No analyzed data available. Please run the analysis first.")

if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection System Dashboard", layout="wide")
    app()

