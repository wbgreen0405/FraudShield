# approval_system_page.py

import streamlit as st

def approval_system_page():
    st.title('Approval System')

    # Fetch the transactions that have been flagged by the model
    flagged_transactions = st.session_state.get('flagged_transactions', None)

    if flagged_transactions is not None:
        st.subheader('Transactions Pending Approval')
        for index, transaction in flagged_transactions.iterrows():
            with st.expander(f"Transaction ID: {transaction['ref_id']}"):
                st.json(transaction)  # Display the transaction details as JSON or use st.write for a DataFrame
                approval = st.radio(
                    "Approve this transaction?", ('Approve', 'Mark as Fraudulent'), key=f"approval_{transaction['ref_id']}"
                )
                if st.button("Submit Decision", key=f"submit_{transaction['ref_id']}"):
                    # Logic to record the approval decision
                    st.session_state['approval_decisions'][transaction['ref_id']] = approval
                    st.success(f"Decision for transaction ID {transaction['ref_id']} recorded as {approval}.")

        # Display a list of recently approved transactions
        approved_transactions = [t for t, decision in st.session_state.get('approval_decisions', {}).items() if decision == 'Approve']
        if approved_transactions:
            st.subheader('Recently Approved Transactions')
            st.write(approved_transactions)  # Replace with a DataFrame or more detailed view
    else:
        st.write("No flagged transactions are available for approval at the moment.")


if __name__ == '__main__':
    approval_system_page()
