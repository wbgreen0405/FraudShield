# Transactions page script
from st_aggrid import AgGrid
from utils.supabase_ops import fetch_transactions
import streamlit as st

def transactions_page():
    st.title('Transactions')
    transactions_data = fetch_transactions()
    if not transactions_data.empty:
        AgGrid(transactions_data)
    else:
        st.write("No transactions data to display.")

