import streamlit as st
import pandas as pd
import plotly.express as px

def app():
    st.title("Anomaly Detection System Dashboard")

    if 'anomaly_df' in st.session_state:
        anomaly_df = st.session_state['anomaly_df']
        
        # Assuming the anomaly detection dataframe has 'ref_id' and some scores or features to plot
        fig = px.scatter(anomaly_df, x='feature_1', y='feature_2', size='anomaly_score', color='anomaly_score',
                         hover_data=['ref_id'], title="Outlier Detection Visualization")
        st.plotly_chart(fig)

        # Display the DataFrame with transactions considered outliers
        st.write("Transactions Identified as Outliers:")
        st.dataframe(anomaly_df[['ref_id', 'feature_1', 'feature_2', 'anomaly_score']], use_container_width=True)
    else:
        st.error("No anomaly detection data available. Please run the analysis first.")

if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection System Dashboard", layout="wide")
    app()
