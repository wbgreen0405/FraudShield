import streamlit as st

def expert_human_judgment_page():
    st.set_page_config(layout="wide")
    st.title('Expert Human Judgment')

    # Check if the flag to display the table is set in session_state
    if hasattr(st.session_state, 'display_combined_flags_table') and st.session_state.display_combined_flags_table:
        # Access the "Combined Flags Table" from session_state
        combined_flags_table = st.session_state.get('combined_flags', None)
        if combined_flags_table is not None:
            st.write("Combined Flags and Anomaly Detection Table:")
            st.write(combined_flags_table.rename(columns={'model_version': 'model_type', 'prob_score': 'score'}))
    else:
        st.info("No transactions available for expert human judgment.")

if __name__ == '__main__':
    expert_human_judgment_page()


