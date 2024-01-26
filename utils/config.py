# Configuration settings for the Streamlit app
import streamlit as st
from supabase import create_client

# Initialize Supabase client using Streamlit secrets
supabase = create_client(
    st.secrets["supabase_url"],
    st.secrets["supabase_key"]
)
