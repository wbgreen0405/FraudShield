import streamlit as st
from pages.home import home_page
from pages.transactions import transactions_page
from pages.offline_review import offline_review_page
from pages.supervised_fraud_results import supervised_fraud_results_page

def render_sidebar():
    # Get the directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the image for the logo in the sidebar
    logo_path = os.path.join(current_dir, 'assets', 'logo.png')
    
    # Display the logo at the top of the sidebar
    st.sidebar.image(logo_path, use_column_width=True)
    
    # Title for the navigation sidebar
    st.sidebar.title('Navigation')
    
    # Dictionary mapping page names to their respective functions
    pages = {
        'Home': home_page,
        'Transactions': transactions_page,
        'Supervised Fraud Results': supervised_fraud_results_page,
        'Offline Review': offline_review_page,
    }

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", list(pages.keys()))
    return pages[page]


def main():
    # Render the sidebar and get the selected page function
    page_func = render_sidebar()
    
    # Call the function to render the selected page
    page_func()

if __name__ == '__main__':
    main()

