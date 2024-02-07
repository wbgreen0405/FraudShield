import streamlit as st
import pandas as pd
import plotly.express as px
import random

# Assuming review_df is generated or loaded in your transactions.py and stored in session_state

def simulate_offline_review():
    """
    Simulates the offline review process and updates the review_df in session_state.
    """
    if 'review_df' in st.session_state:
        review_df = st.session_state['review_df']
        # Simulation logic here
        for index, row in review_df.iterrows():
            # Simplified simulation logic for illustration
            review_df.at[index, 'expert_decision'] = random.choice(['Confirmed Fraud', 'Confirmed Legitimate'])
        st.session_state['review_df'] = review_df

def plot_workflow_diagram():
    """
    Plots a workflow diagram showing the distribution of expert decisions.
    """
    if 'review_df' in st.session_state:
        fig = px.pie(st.session_state['review_df'], names='expert_decision', title='Review Cases Status')
        st.plotly_chart(fig)

def plot_case_resolution_timeline(review_df):
    """
    Plots a case resolution timeline using start and end times from review_df.
    """
    # Ensure the data has the expected columns
    if 'review_start' in review_df.columns and 'review_end' in review_df.columns and 'ref_id' in review_df.columns:
        # Convert review_start and review_end to datetime if they're not already
        review_df['review_start'] = pd.to_datetime(review_df['review_start'])
        review_df['review_end'] = pd.to_datetime(review_df['review_end'])
        
        # Plotly Express timeline requires a start, end, and a category (y-axis) for each event
        fig = px.timeline(review_df, x_start='review_start', x_end='review_end', y='ref_id', labels={'ref_id': 'Case ID'})
        
        # Update layout for better readability
        fig.update_layout(xaxis_title='Time', yaxis_title='Case ID', title='Case Resolution Timeline')
        fig.update_yaxes(categoryorder='total ascending')  # This ensures the timeline is sorted by case ID
        
        st.plotly_chart(fig)
    else:
        st.error("Dataframe does not contain the required columns for plotting the timeline.")

# Example usage, assuming review_df is available in the session state
if 'review_df' in st.session_state:
    plot_case_resolution_timeline(st.session_state['review_df'])
else:
    st.error("Review data not found.")

def show_case_detail(case_id):
    """
    Shows details for a selected case.
    """
    if 'review_df' in st.session_state:
        case_data = st.session_state['review_df'][st.session_state['review_df']['ref_id'] == case_id]
        if not case_data.empty:
            st.write(case_data)
        else:
            st.error("Case not found!")

def app():
    st.title("Expert Review Dashboard")
    
    # Simulate Offline Review button
    if st.button('Simulate Offline Review'):
        simulate_offline_review()
        st.success("Simulation complete. Expert decisions have been added.")

    # Workflow Diagram
    plot_workflow_diagram()

    # Make sure to check if 'review_df' is in st.session_state before calling the plot functions
    if 'review_df' in st.session_state:
        # Now pass the 'review_df' from st.session_state to the function
        plot_case_resolution_timeline(st.session_state['review_df'])

        # Select a case to review in detail
        case_id_option = st.selectbox("Select a case to review in detail:", st.session_state['review_df']['ref_id'].unique())
        show_case_detail(case_id_option)

        # Display updated dataframe after Expert Review
        st.subheader("Updated Transactions after Expert Review")
        st.dataframe(st.session_state['review_df'])

if __name__ == '__main__':
    app()

