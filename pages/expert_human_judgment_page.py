import streamlit as st
import pandas as pd
import datetime
import random
from pages.transactions import log_audit_entry, unified_flags, anomaly_detection_records  # Add these import statements

def expert_human_judgment_page():
    st.set_page_config(layout="wide")
    st.title('Expert Human Judgment')

    # Directly check and display the Combined Flags Table if it exists in session_state
    if 'combined_flags_table' in st.session_state:
        combined_flags_table = st.session_state['combined_flags_table']
        st.write("Combined Flags and Anomaly Detection Table:")
        st.dataframe(combined_flags_table.rename(columns={'model_version': 'model_type', 'prob_score': 'score'}))
    else:
        st.info("No Combined Flags and Anomaly Detection Table available.")

if __name__ == '__main__':
    expert_human_judgment_page()
