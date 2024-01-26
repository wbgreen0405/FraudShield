# The entry point for the Streamlit app
import os
import streamlit as st
from pages.home import home_page
from pages.transactions import transactions_page

def render_sidebar():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(current_dir, 'assets', 'logo.png')
    st.sidebar.image(logo_path, use_column_width=True)
    st.sidebar.title('Navigation')
    pages = {
        'Home': home_page,
        'Transactions': transactions_page
        # Add other pages as needed
    }
    page = st.sidebar.radio('Select a page:', list(pages.keys()))
    return pages[page]

def main():
    page_func = render_sidebar()    
    page_func()

if __name__ == '__main__':
    main()


