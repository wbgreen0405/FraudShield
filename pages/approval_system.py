import streamlit as st

def app():
    # Assuming 'approval_system_df' is already set in the session state by the main analysis function
    if 'supervised_df' in st.session_state:
        approval_df = st.session_state['supervised_df']
        
        # Ensure 'rf_predicted_fraud' column is present
        if 'rf_predicted_fraud' in supervised_df.columns:
            # Optionally, map numeric predictions to string labels for clarity
            supervised_df['Fraud Status'] = supervised_df['rf_predicted_fraud'].map({1: 'Fraud', 0: 'Not Fraud'})
            
            # Display the DataFrame with an emphasis on the Fraud Status
            st.write("Approval System Results:")
            st.dataframe(supervised_df[['ref_id', 'Fraud Status']], use_container_width=True)  # Use 'ref_id' as the transaction identifier
            
        else:
            st.error("The 'rf_predicted_fraud' column is missing from the approval system data.")
    else:
        st.error('Approval system data not found.')

# This line is optional, used for direct script testing
if __name__ == '__main__':
    st.set_page_config(page_title="Approval System", layout="wide")
    app()



