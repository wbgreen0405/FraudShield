import streamlit as st
import pandas as pd
import plotly.express as px
import random
from datetime import datetime, timedelta


def simulate_offline_review(review_df):
    # Ensure review_df is not None
    if review_df is None:
        return None

    # Simulate review dates if not already present
    if 'review_start' not in review_df.columns or 'review_end' not in review_df.columns:
        review_start_dates = []
        review_end_dates = []
        for _ in range(len(review_df)):
            start_date = datetime.now() - timedelta(days=random.randint(1, 30))
            end_date = start_date + timedelta(hours=random.randint(1, 48))
            review_start_dates.append(start_date)
            review_end_dates.append(end_date)
        review_df['review_start'] = review_start_dates
        review_df['review_end'] = review_end_dates

    # Simulate expert decisions if not already present
    if 'expert_decision' not in review_df.columns:
        decisions = ['Confirmed Fraud', 'Confirmed Legitimate']
        review_df['expert_decision'] = [random.choice(decisions) for _ in review_df.index]

    return review_df


def plot_workflow_diagram(review_df):
    if 'expert_decision' in review_df.columns:
        fig = px.pie(review_df, names='expert_decision', title='Review Cases Status')
        st.plotly_chart(fig)
    else:
        st.warning("Run the simulation to generate expert decisions.")

def plot_case_resolution_timeline(review_df):
    required_cols = ['review_start', 'review_end', 'ref_id', 'expert_decision']
    if all(col in review_df.columns for col in required_cols):
        review_df['review_start'] = pd.to_datetime(review_df['review_start'], errors='coerce')
        review_df['review_end'] = pd.to_datetime(review_df['review_end'], errors='coerce')
        fig = px.timeline(review_df, x_start='review_start', x_end='review_end', y='ref_id', color='expert_decision', labels={'ref_id': 'Case ID', 'expert_decision': 'Decision'})
        fig.update_layout(xaxis_title='Time', yaxis_title='Case ID', title='Case Resolution Timeline')
        fig.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig)
    else:
        st.warning("Review data does not contain the required columns for plotting the timeline.")

def show_case_detail(review_df, case_id):
    case_data = review_df[review_df['ref_id'] == case_id]
    if not case_data.empty:
        st.write(case_data)
    else:
        st.error("Case not found!")

def app():
    
    st.title("Expert Review Dashboard")
    
    # Check if the transaction analysis has been completed
    if 'transaction_analysis_completed' in st.session_state and st.session_state['transaction_analysis_completed']:
        if 'review_df' in st.session_state and st.session_state['review_df'] is not None:
            review_df = st.session_state['review_df']

            # Drop unnecessary columns
            columns_to_drop = ['RF Approval Status', 'LOF Status', 'LOF Status_x', 'rf_predicted_fraud', 'LOF Status_y', 'lof_scores_y']
            review_df = review_df.drop(columns=columns_to_drop, errors='ignore')

            # Automatically simulate offline review if not done yet
            if 'offline_review_simulated' not in st.session_state:
                review_df = simulate_offline_review(review_df)
                st.session_state['review_df'] = review_df  # Update review_df in session state
                st.session_state['offline_review_simulated'] = True  # Mark simulation as done
                st.success("Offline review simulation complete. Expert decisions have been added.")

            # Visualization and detailed review sections
            col1, col2 = st.columns(2)
            with col1:
                plot_workflow_diagram(review_df)
            with col2:
                plot_case_resolution_timeline(review_df)
            case_id_option = st.selectbox("Select a case to review in detail:", review_df['ref_id'].unique())
            show_case_detail(review_df, case_id_option)
            st.subheader("Updated Transactions after Expert Review")
            st.dataframe(review_df)
        else:
            st.error("No transaction data available for review. Please analyze transactions first.")
    else:
        st.error("Please complete the transaction analysis before proceeding to the expert review dashboard.")

app()
