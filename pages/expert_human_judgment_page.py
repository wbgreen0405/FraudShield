import streamlit as st
import pandas as pd
import datetime
import random
from pages.transactions import log_audit_entry, unified_flags, anomaly_detection_records  # Add these import statements

def expert_human_judgment_page():
    st.title("Expert Human Judgment")

    # Debug statement to check the value of display_combined_flags_table
    print("Display Combined Flags Table:", st.session_state.get('display_combined_flags_table', None))

    # Check if the flag to display the table is set in session_state
    if hasattr(st.session_state, 'display_combined_flags_table') and st.session_state.display_combined_flags_table:
        # Debug statement to check if we enter this block
        print("Displaying Combined Flags Table")
        
        # Display the "Combined Flags and Anomaly Detection Table" here
        combined_flags_table = st.session_state.get('combined_flags_table', None)
        if combined_flags_table is not None:
            st.write("Combined Flags and Anomaly Detection Table:")
            st.write(combined_flags_table)
    else:
        st.info("No transactions available for expert human judgment.")

if __name__ == '__main__':
    expert_human_judgment_page()
