import streamlit as st
import pandas as pd
import plotly.express as px

def create_anomaly_detection_plot(full_df):
    # Normalize LOF scores for visualization
    full_df['lof_scores_normalized'] = (full_df['lof_scores'] - full_df['lof_scores'].min()) / (full_df['lof_scores'].max() - full_df['lof_scores'].min())
    
    # Assign 'Outlier Status' based on LOF predictions
    full_df['Outlier Status'] = full_df['lof_predicted_fraud'].map({-1: 'Outlier', 1: 'Inlier'})

    # Create scatter plot
    fig = px.scatter(
        full_df, 
        x='ref_id', 
        y='lof_scores_normalized', 
        color='Outlier Status',  # Color by Outlier Status for clarity
        title="Anomaly Detection Scatter Plot",
        hover_data=['income', 'name_email_similarity']  # Ensure 'lof_scores' is included in your dataframe
    )
    
    # Update layout if needed
    fig.update_layout(xaxis_title="Reference ID", yaxis_title="LOF Scores Normalized")

    return fig

# Replace the placeholders with your actual data fetching and inference code
# This is just a placeholder to simulate loading your dataset
def load_data():
    # Load your full dataset here
    return pd.DataFrame()

def app():
    st.title("Anomaly Detection System Dashboard")

    # Assuming you've run your analysis and have 'lof_scores' and 'lof_predicted_fraud' in your full dataset
    full_df = load_data()

    # Check if data is loaded
    if not full_df.empty:
        # Plot scatter plot based on the full dataset
        st.subheader("Anomaly Scatter Plot")
        fig = create_anomaly_detection_plot(full_df)
        st.plotly_chart(fig)

        # Display detailed transactions (optional, depending on your needs)
        st.subheader("Detailed Transactions")
        st.dataframe(full_df[['ref_id', 'lof_scores']], use_container_width=True)
    else:
        st.error("No data available. Please load the data first.")

if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection System Dashboard", layout="wide")
    app()

