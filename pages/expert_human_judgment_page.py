import streamlit as st
import pandas as pd
import datetime
import random

# Define an empty list to store audit logs
audit_logs = []

# Function to log an entry in the audit logs
def log_audit_entry(transaction_id, reviewer_id, decision):
    timestamp = datetime.datetime.now()
    audit_entry = {
        'Timestamp': timestamp,
        'Transaction ID': transaction_id,
        'Reviewer ID': reviewer_id,
        'Decision': decision,
    }
    audit_logs.append(audit_entry)

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

    # Create a DataFrame to display the simulated human review decisions
    human_review_records = []
    
    # Create a button to trigger the simulation
    if st.button("Simulate Review"):
        for index in offline_review_transactions:
            transaction = transactions_data.iloc[index]
            original_decision = st.session_state.get(f'original_decision_{index}', 'N/A')
            
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

            # Store the original decision in session_state
            st.session_state[f'original_decision_{index}'] = updated_decision

            # Add a row to human_review_records
            row = {
                'Transaction ID': index,
                'Original Decision': original_decision,
                'Updated Decision': updated_decision,
                'Model Type': 'Simulated',
                'Reviewer ID': 'simulated_reviewer',
                'Reviewed At': datetime.datetime.now(),
                'Comments': 'Simulated review decision',
            }
            human_review_records.append(row)

    # Create a DataFrame from human_review_records
    human_review_df = pd.DataFrame(human_review_records)

    # Apply background color to the DataFrame
    def background_color(s):
        return ['background-color: #FFC000' if s['Original Decision'] != s['Updated Decision'] else '' for _ in s]

    human_review_df_styled = human_review_df.style.apply(background_color, axis=1)

    # Display the human review decisions in a table
    st.write(human_review_df_styled)

if __name__ == '__main__':
    expert_human_judgment_page()



