import streamlit as st
import pandas as pd

# Define your expert_human_judgment_page function
def expert_human_judgment_page():
    st.set_page_config(layout="wide")
    st.title('Expert Human Judgment')

    # Debug: Print the session state to verify the contents
    st.write("Session State:", st.session_state)

    # Check if 'combined_flags_table' exists in session_state
    if 'combined_flags_table' in st.session_state:
        combined_flags_table = st.session_state['combined_flags_table']
        
        # Ensure it's a DataFrame
        if isinstance(combined_flags_table, pd.DataFrame):
            st.write("Combined Flags and Anomaly Detection Table:")
            
            # Display the DataFrame as a table
            st.dataframe(combined_flags_table)
        else:
            st.write("The combined flags table is not in the expected format.")
    else:
        st.info("No Combined Flags and Anomaly Detection Table available.")

# Check if the script is executed as the main program
if __name__ == '__main__':
    expert_human_judgment_page()
