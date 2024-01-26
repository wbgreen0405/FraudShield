# Operations related to Supabase
import pandas as pd
from utils.config import supabase
import streamlit as st

def fetch_transactions():
    data = supabase.table('transactions_table_name').select('*').execute()
    if data.error:
        st.error('Failed to retrieve data from Supabase: ' + str(data.error))
        return pd.DataFrame()
    return pd.DataFrame(data.data)
