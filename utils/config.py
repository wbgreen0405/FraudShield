# Configuration settings for the Streamlit app
import streamlit as st
from supabase import create_client


# Initialize Supabase client using Streamlit secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
# The create_client function should be imported from wherever you have it defined,
# for example from the supabase-py package or a custom module.
supabase = create_client(supabase_url, supabase_key)
