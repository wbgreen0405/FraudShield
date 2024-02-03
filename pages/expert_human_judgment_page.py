import streamlit as st
import pandas as pd
import datetime
import random
# Assuming the import path is correct and the specified imports are necessary for your app context
from pages.transactions import log_audit_entry, unified_flags, anomaly_detection_records

def expert_human_judgment_page():
    st.set_page_config(layout="wide")
    st.title('Expert Human Judgment')

    # Debug: Print the session state to verify the contents
    st.write("Session State:", st.session_state)

    # Check if 'combined_flags_table' exists in session_state and is not None
    if 'combined_flags_table' in st.session_state and st.session_state['combined_flags_table'] is not None:
        combined_flags_table = st.session_state['combined_flags_table']
        # Ensure the table is a DataFrame before attempting to rename and display
        if isinstance(combined_flags_table, pd.DataFrame):
            st.write("Combined Flags and Anomaly Detection Table:")
            # Ensure the DataFrame has the columns you're attempting to rename
            if set(['model_version', 'prob_score']).issubset(combined_flags_table.columns):
                st.dataframe(combined_flags_table.rename(columns={'model_version': 'model_type', 'prob_score': 'score'}))
            else:
                st.write("Note: Not all columns to rename exist in the DataFrame.")
        else:
            st.write("The combined flags table is not in the expected format.")
    else:
        st.info("No Combined Flags and Anomaly Detection Table available.")

if __name__ == '__main__':
    expert_human_judgment_page()

