import streamlit as st
import pandas as pd
import plotly.express as px

# Replace 'income' and 'name_email_similarity' with actual numerical features from your dataset
def create_anomaly_detection_plot(anomaly_df):
    fig = px.scatter(
        anomaly_df, 
        x='income',  # Replace with your actual feature name
        y='name_email_similarity',  # Replace with your actual feature name
        color='rf_predicted_fraud',  # This will color the points by their fraud status
        title="Anomaly Detection Scatter Plot",
        labels={
            'rf_predicted_fraud': 'Fraud Status'
        },
        hover_data=['ref_id', 'lof_scores']  # Showing the 'ref_id' and 'lof_scores' on hover
    )
    return fig

def app():
    st.title("Anomaly Detection System Dashboard")

    if 'anomaly_df' in st.session_state:
        anomaly_df = st.session_state['anomaly_df']

        # Plot the scatter plot
        st.subheader("Anomaly Scatter Plot")
        fig = create_anomaly_detection_plot(anomaly_df)
        st.plotly_chart(fig)

        # Display the table of transactions considered as anomalies
        st.subheader("Detailed Anomaly Transactions")
        st.dataframe(anomaly_df[['ref_id', 'income', 'name_email_similarity', 'lof_scores']], use_container_width=True)
    else:
        st.error("No anomalies found. Run the analysis to detect anomalies.")

if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection System Dashboard", layout="wide")
    app()
