import streamlit as st
import pandas as pd
import plotly.express as px

# Function to add 'Outlier Status' based on 'LOF Status'
def add_outlier_status(df):
    # Ensure 'LOF Status' is present and map it to 'Outlier Status'
    if 'LOF Status' in df.columns:
        df['Outlier Status'] = df['LOF Status'].map({'Suspected Fraud': 'Outlier', None: 'Inlier', 'Non-Fraud': 'Inlier'})
    else:
        df['Outlier Status'] = 'Inlier'  # Default to 'Inlier' if 'LOF Status' is not available
    return df

# Function to create anomaly detection scatter plot
def create_anomaly_detection_plot(df):
    # Use 'lof_scores' directly for y-axis values
    fig = px.scatter(
        df,
        x='ref_id',
        y='lof_scores',  # Assuming 'lof_scores' column exists and is populated
        color='Outlier Status',  # Use the 'Outlier Status' column for color
        title="Anomaly Detection Scatter Plot",
        hover_data=['ref_id', 'lof_scores']
    )
    fig.update_layout(xaxis_title="Suspected Fraud Transactions", yaxis_title="LOF Scores")
    return fig

# Function to create LOF scores distribution plot
def create_lof_distribution_plot(df):
    fig = px.histogram(
        df,
        x='lof_scores',  # Directly use 'lof_scores' for the x-axis
        nbins=20,
        title="Distribution of LOF Scores"
    )
    return fig

# Streamlit app function
def app():
    st.title("Anomaly Detection System Dashboard")

    if 'anomaly_df' in st.session_state:
        non_fraud_df = st.session_state['anomaly_df']
        # Ensure 'lof_scores' is present; you might also normalize or calculate it beforehand
        if 'lof_scores' not in non_fraud_df.columns:
            st.warning("LOF scores are missing in the analyzed data.")
            return

        # Add 'Outlier Status' based on 'LOF Status'
        non_fraud_df = add_outlier_status(non_fraud_df)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Anomaly Scatter Plot")
            scatter_fig = create_anomaly_detection_plot(non_fraud_df)
            st.plotly_chart(scatter_fig)

        with col2:
            st.subheader("LOF Scores Distribution")
            dist_fig = create_lof_distribution_plot(non_fraud_df)
            st.plotly_chart(dist_fig)

        # Filter for outliers based on 'Outlier Status'
        outliers_df = non_fraud_df[non_fraud_df['Outlier Status'] == 'Outlier']

        st.subheader("Detailed Outlier Transactions")
        st.dataframe(outliers_df, use_container_width=True)
    else:
        st.error("No analyzed data available. Please run the analysis first.")

#if __name__ == '__main__':
app()


