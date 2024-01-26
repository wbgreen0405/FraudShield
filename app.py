import os
import streamlit as st

# Get the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the image for the logo in the sidebar
logo_path = os.path.join(current_dir, 'assets', 'logo_transparent.png')  # Update the path if needed

# Function to render the sidebar with the logo and navigation
def render_sidebar():
    st.sidebar.image(logo_path, use_column_width=True)  # Display the logo at the top of the sidebar
    st.sidebar.title('Navigation')
    pages = {
        'Home': home_page,
        'Data Upload and Preprocessing': data_upload_page,  # Placeholder function for now
        'Model Predictions': model_predictions_page,        # Placeholder function for now
        'Anomaly Detection': anomaly_detection_page,        # Placeholder function for now
        'Human Judgment Simulation': human_judgment_page,   # Placeholder function for now
        'Review and Update': review_update_page             # Placeholder function for now
    }
    # Radio button for page navigation
    page = st.sidebar.radio('Select a page:', list(pages.keys()))
    # Return the selected page function
    return pages[page]

# Function to render the home page content based on the slide
def home_page():
    st.title('Welcome to FraudShield: The Fraud Detection System Demo.')
    st.markdown("""
    Dive into the world of anti-fraud analytics with our streamlined, user-friendly platform designed to highlight suspicious financial activities through data-driven insights. FraudShield provides a comprehensive suite of tools to identify, analyze, and combat fraudulent transactions, ensuring your financial security.

    **Key Features:**
    - Data Upload and Preprocessing: Securely upload and preprocess your transactional data.
    - Model Predictions: Use a pre-trained Random Forest classifier to identify potential fraud cases.
    - Anomaly Detection: Apply the Local Outlier Factor model to scrutinize flagged transactions.
    - Human Judgment Simulation: Simulate expert review decisions to validate flagged cases.
    - Review and Update: Update the dataset based on the review outcomes and prepare it for further analysis or model retraining.

    **How to Use:** Navigate through the sidebar to access different stages of the fraud detection process. Start by uploading your dataset and then proceed through each stage to explore the full capabilities of the system.

    Inspired by innovative methodologies in fraud detection, this app offers an in-depth look at how data science can be leveraged to protect financial assets and maintain transactional integrity.
    """)

    # Add a button or any other elements if needed here

# Placeholder functions for other pages
def data_upload_page():
    st.title('Data Upload and Preprocessing')
    # Add content for data upload and preprocessing

def model_predictions_page():
    st.title('Model Predictions')
    # Add content for model predictions

def anomaly_detection_page():
    st.title('Anomaly Detection')
    # Add content for anomaly detection

def human_judgment_page():
    st.title('Human Judgment Simulation')
    # Add content for human judgment simulation

def review_update_page():
    st.title('Review and Update')
    # Add content for review and update

# Main function to set up the Streamlit app structure
def main():
    # Render the sidebar and get the selected page function
    page_func = render_sidebar()
    
    # Render the selected page
    page_func()

if __name__ == '__main__':
    main()

