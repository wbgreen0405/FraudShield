import streamlit as st
import pandas as pd
import datetime
import random
from pages.transactions import log_audit_entry


def expert_human_judgment_page():
    st.title("Expert Human Judgment")

    # Access the offline_review_transactions from session_state
    offline_review_transactions = st.session_state.get('offline_review_transactions', [])

    if not offline_review_transactions:
        st.info("No transactions available for expert human judgment.")
        return

    st.write("Here you can simulate the expert human judgment process.")

    # Retrieve transactions_data from session_state
    transactions_data = st.session_state.get('transactions_data', None)

    if transactions_data is None:
        st.error("Transactions data is missing. Please run preprocessing and inference first.")
        return

    # Retrieve potential fraud and LOF anomaly indices from session_state
    potential_fraud_indices = st.session_state.get('potential_fraud_indices', [])
    lof_anomaly_indices = st.session_state.get('lof_anomaly_indices', [])

    # Combine LOF anomalies and RF frauds for human review
    offline_review_transactions = set(potential_fraud_indices + lof_anomaly_indices)
    st.session_state['offline_review_transactions'] = list(offline_review_transactions)

    # Initialize original_decision
    original_decision = "Possible Fraud"  # Replace with the actual original decision
    model_type = "LOF"  # Replace with the correct model type based on the data

    # Create a button to trigger the simulation
    simulate_button = st.button("Simulate Review")  # Define the simulate_button variable here

    if not simulate_button:
        # Display the original table with both RF_v1 and LOF_v1
        human_review_records = []
        for index in offline_review_transactions:
            transaction = transactions_data.iloc[index]
            original_decision_rf = None
            original_decision_lof = None
    
            # Find the corresponding entry in possible_frauds
            for fraud in possible_frauds:
                if fraud['flag_id'] == transaction['ref_id'] and fraud['model_version'] == 'RF_v1':
                    original_decision_rf = 'possible fraud'
    
            # Find the corresponding entry in anomalies
            for anomaly in anomalies:
                if anomaly['anomaly_id'] == transaction['ref_id'] and anomaly['model_version'] == 'LOF_v1':
                    original_decision_lof = 'possible fraud'
    
            row = {
                'Transaction ID': index,
                'Original Decision (RF_v1)': original_decision_rf,
                'Original Decision (LOF_v1)': original_decision_lof,
            }
            human_review_records.append(row)
    
        # Create a DataFrame for the combined human review results
        combined_human_review_df = pd.DataFrame(human_review_records)
    
        # Fill any empty LOF_v1 decisions with "Not Reviewed"
        combined_human_review_df['Original Decision (LOF_v1)'].fillna("Not Reviewed", inplace=True)
    
        # Display the original table with both RF_v1 and LOF_v1
        st.write("Before (Both RF_v1 and LOF_v1):")
        st.write(combined_human_review_df)
    else:
        # Clear the current content on the page
        st.empty()

        # Create a DataFrame to display the simulated human review decisions
        simulated_human_review_records = []
        for index in offline_review_transactions:
            transaction = transactions_data.iloc[index]

            # Function to simulate offline review
            def simulate_offline_review(transaction):
                INCOME_THRESHOLD = 100000
                AGE_THRESHOLD = 50
                EMPLOYMENT_STATUS_SUSPICIOUS = 3
                HOUSING_STATUS_SUSPICIOUS = 2
                ERROR_RATE = 0.1

                is_unusually_high_income = transaction['income'] > INCOME_THRESHOLD
                is_age_above_threshold = transaction['customer_age'] > AGE_THRESHOLD
                is_suspicious_employment = transaction['employment_status'] == EMPLOYMENT_STATUS_SUSPICIOUS
                is_suspicious_housing = transaction['housing_status'] == HOUSING_STATUS_SUSPICIOUS

                if is_unusually_high_income and (is_age_above_threshold or is_suspicious_employment or is_suspicious_housing):
                    decision = 'fraudulent'
                else:
                    decision = 'legitimate'

                if random.random() < ERROR_RATE:
                    decision = 'legitimate' if decision == 'fraudulent' else 'fraudulent'

                # Log an entry in the audit logs for this decision
                log_audit_entry(transaction_id=index, reviewer_id='simulated_reviewer', decision=decision)

                return decision

            # Simulate offline review for the transaction
            updated_decision = simulate_offline_review(transaction)

            # Add a row to simulated_human_review_records
            row = {
                'Transaction ID': index,
                'Original Decision (RF_v1)': None,
                'Original Decision (LOF_v1)': None,
                'Updated Decision': updated_decision,
            }
            simulated_human_review_records.append(row)

        # Create a DataFrame for the simulated human review results
        simulated_human_review_df = pd.DataFrame(simulated_human_review_records)

        # Apply background color to changed rows
        def highlight_changed_cells(s):
            changed_rows = s['Original Decision (RF_v1)'] != s['Updated Decision']
            df = pd.DataFrame('', index=s.index, columns=s.columns)
            df.loc[changed_rows, :] = 'background-color: #FFC000'
            return df

        # Apply the highlight function to the DataFrame
        styled_simulated_human_review_df = simulated_human_review_df.style.apply(highlight_changed_cells, axis=None)

        # Display the simulated human review decisions in a table with background color
        st.write("After (Both RF_v1 and LOF_v1):")
        st.write(styled_simulated_human_review_df)

if __name__ == '__main__':
    expert_human_judgment_page()
