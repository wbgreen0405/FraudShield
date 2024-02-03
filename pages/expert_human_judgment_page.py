import streamlit as st

def expert_human_judgment_page():
    st.set_page_config(layout="wide")
    st.title('Expert Human Judgment')

    # Check if the flag to display the table is set in session_state
    if 'combined_flags_table' in st.session_state:
        # Access the "Combined Flags Table" from session_state
        combined_flags_table = st.session_state.combined_flags_table
        if combined_flags_table is not None:
            st.write("Combined Flags and Anomaly Detection Table:")
            st.write(combined_flags_table.rename(columns={'model_version': 'model_type', 'prob_score': 'score'}))
    else:
        st.info("No transactions available for expert human judgment.")

if __name__ == '__main__':
    expert_human_judgment_page()


