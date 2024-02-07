import streamlit as st
import pandas as pd
import plotly.express as px

# Function to add 'Outlier Status' to the DataFrame
def add_outlier_status(df):
    if 'lof_predicted_fraud' in df.columns:
        df['Outlier Status'] = df['lof_predicted_fraud'].map({-1: 'Outlier', 1: 'Inlier'})
    return df

# Function to create anomaly detection scatter plot
def create_anomaly_detection_plot(df):
    fig = px.scatter(
        df,
        x='ref_id',
        y='lof_scores_normalized',
        color='Outlier Status',  # Assumes 'Outlier Status' column exists
        title="Anomaly Detection Scatter Plot",
        hover_data=['ref_id', 'lof_scores_normalized']
    )
    fig.update_layout(xaxis_title="Reference ID", yaxis_title="Normalized LOF Scores")
    return fig

# Function to create LOF scores distribution plot
def create_lof_distribution_plot(df):
    fig = px.histogram(
        df,
        x='lof_scores_normalized',
        nbins=20,
        title="Distribution of Local Outlier Factor Scores"
    )
    return fig

# Streamlit app function
def app():
    st.title("Anomaly Detection System Dashboard")

    if 'analyzed_df' in st.session_state:
        analyzed_df = st.session_state['analyzed_df']

        # Add 'Outlier Status' to the DataFrame
        analyzed_df = add_outlier_status(analyzed_df)

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

        # Filter for outliers and prepare detailed transactions view
        outliers_df = analyzed_df[analyzed_df['Outlier Status'] == 'Outlier']
        
        # Drop unwanted columns
        columns_to_drop = ['rf_prob_scores', 'rf_predicted_fraud', 'Approval Status']
        outliers_df = outliers_df.drop(columns=[col for col in columns_to_drop if col in outliers_df.columns], errors='ignore')

        # Reorder columns
        cols = ['ref_id', 'lof_scores', 'lof_scores_normalized'] + [col for col in outliers_df.columns if col not in ['ref_id', 'lof_scores', 'lof_scores_normalized']]
        outliers_df = outliers_df[cols]

        st.subheader("Detailed Anomaly Transactions")
        st.dataframe(outliers_df, use_container_width=True)
    else:
        st.error("No analyzed data available. Please run the analysis first.")

if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection System Dashboard", layout="wide")
    app()
