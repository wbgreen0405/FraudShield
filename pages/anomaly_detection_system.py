import streamlit as st
from st_aggrid import AgGrid
from utils.supabase_ops import fetch_anomaly_detection_records
from utils.chart_helpers import plot_lof_outlier_detection

def anomaly_detection_system_page():
    st.title('Anomaly Detection System')

    # Fetch anomaly detection records from Supabase or your data source
    anomaly_records = fetch_anomaly_detection_records()

    # Display LOF Outlier Detection chart
    st.subheader('LOF Outlier Detection')
    if not anomaly_records.empty:
        lof_chart = plot_lof_outlier_detection(anomaly_records['lof_score'])
        st.pyplot(lof_chart)

        # Display the table with suspect fraud records
        st.subheader('Suspect Fraud')
        AgGrid(anomaly_records)
    else:
        st.write("No anomaly detection records to display.")

# Run this page function
anomaly_detection_system_page()
