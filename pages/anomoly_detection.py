import streamlit as st
import pandas as pd
import plotly.express as px

def create_anomaly_detection_plot(analyzed_df):
    # Filter for Non-Fraud transactions
    non_fraud_df = analyzed_df[analyzed_df['Approval Status'] == 'Non-Fraud'].copy()

    
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
    non_fraud_df = analyzed_df[analyzed_df['Approval Status'] == 'Non-Fraud'].copy()


    fig = px.histogram(
        non_fraud_df,
        x='lof_scores_normalized',
        nbins=20,  # Adjust the number of bins as needed
        title="Distribution of Local Outlier Factor Scores"
    )
    return fig

def remove_duplicate_columns(df):
    # Create an empty set for column names
    cols_seen = set()
    # Create an empty list for columns to keep
    cols_to_keep = []

    for col in df.columns:
        # If the column is in the set, it's a duplicate, so we modify it
        if col in cols_seen:
            col = f"{col}_duplicate"
        # Add the original or modified column to the list and set
        cols_to_keep.append(col)
        cols_seen.add(col)

    # Select the columns to keep from the dataframe
    return df[cols_to_keep]

def app():
    st.title("Anomaly Detection System Dashboard")

    if 'analyzed_df' in st.session_state:
        analyzed_df = st.session_state['analyzed_df']

        # Remove duplicate columns before plotting
        analyzed_df = remove_duplicate_columns(analyzed_df)

        col1, col2 = st.columns(2)

        with col1:
            # Plot scatter plot based on analyzed_df
            st.subheader("Anomaly Scatter Plot")
            scatter_fig = create_anomaly_detection_plot(analyzed_df)
            st.plotly_chart(scatter_fig)

        with col2:
            # Plot LOF scores distribution
            st.subheader("LOF Scores Distribution")
            dist_fig = create_lof_distribution_plot(analyzed_df)
            st.plotly_chart(dist_fig)

        # Display detailed transactions
        st.subheader("Detailed Anomaly Transactions")

        # Filter for outliers if necessary
        outliers_df = analyzed_df[analyzed_df['Outlier Status'] == 'Outlier']

        # Reorder the columns as needed and exclude unwanted columns
        cols_to_show = ['ref_id', 'lof_scores', 'lof_scores_normalized'] + \
                       [col for col in outliers_df.columns if col not in ['rf_prob_scores', 'rf_predicted_fraud', 'Approval Status', 'Outlier Status']]
        outliers_df = outliers_df[cols_to_show]

        st.dataframe(outliers_df, use_container_width=True)
    else:
        st.error("No analyzed data available. Please run the analysis first.")

if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection System Dashboard", layout="wide")
    app()

