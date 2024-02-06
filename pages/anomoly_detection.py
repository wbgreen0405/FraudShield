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
    # Remove duplicate columns from a DataFrame and append a suffix to duplicates.
    df = df.loc[:, ~df.columns.duplicated()]
    return df

import streamlit as st
import pandas as pd
import plotly.express as px

# Define your functions `create_anomaly_detection_plot` and `create_lof_distribution_plot` here

def app():
    st.title("Anomaly Detection System Dashboard")

    if 'analyzed_df' in st.session_state:
        analyzed_df = st.session_state['analyzed_df']

        # Clean the dataframe of any duplicate columns
        analyzed_df = remove_duplicate_columns(analyzed_df)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Anomaly Scatter Plot")
            scatter_fig = create_anomaly_detection_plot(analyzed_df)
            st.plotly_chart(scatter_fig)

        with col2:
            st.subheader("LOF Scores Distribution")
            dist_fig = create_lof_distribution_plot(analyzed_df)
            st.plotly_chart(dist_fig)

        # Assuming 'Outlier Status' is properly set up in the analyzed_df
        # Display only the outliers in a detailed transactions dataframe
        st.subheader("Detailed Anomaly Transactions")
        outlier_condition = (analyzed_df['Outlier Status'] == 'Outlier')
        detailed_outliers_df = analyzed_df[outlier_condition]

        # Display all columns in the dataframe or specify columns you're interested in
        st.dataframe(detailed_outliers_df, use_container_width=True)
    else:
        st.error("No analyzed data available. Please run the analysis first.")

if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection System Dashboard", layout="wide")
    app()

