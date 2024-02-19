import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import random
import numpy as np

# Assuming other necessary imports and function definitions (load_model_from_s3, fetch_transactions, etc.) are already in place

def simulate_offline_review(review_df):
    if 'review_start' not in review_df.columns or 'review_end' not in review_df.columns:
        review_start_dates, review_end_dates = [], []
        for _ in range(len(review_df)):
            start_date = datetime.now() - timedelta(days=random.randint(1, 30))
            end_date = start_date + timedelta(hours=random.randint(1, 48))
            review_start_dates.append(start_date)
            review_end_dates.append(end_date)
        review_df['review_start'] = review_start_dates
        review_df['review_end'] = review_end_dates

    if 'expert_decision' not in review_df.columns:
        # Set 2% of the decisions to "Confirmed Legitimate"
        probabilities = [0.98, 0.02]  # 98% fraud, 2% legitimate
        choices = ['Confirmed Fraud', 'Confirmed Legitimate']
        decisions = np.random.choice(choices, size=len(review_df), p=probabilities)
        review_df['expert_decision'] = decisions

    return review_df

def plot_workflow_diagram(review_df):
    fig = px.pie(review_df, names='expert_decision', title='Review Cases Status')
    st.plotly_chart(fig)

def plot_case_resolution_timeline(review_df):
    fig = px.timeline(review_df, x_start='review_start', x_end='review_end', y='ref_id', color='expert_decision', labels={'ref_id': 'Case ID', 'expert_decision': 'Decision'})
    fig.update_layout(xaxis_title='Time', yaxis_title='Case ID', title='Case Resolution Timeline')
    st.plotly_chart(fig)

def show_case_detail(review_df, case_id):
    case_data = review_df[review_df['ref_id'] == case_id]
    if not case_data.empty:
        st.write(case_data)
    else:
        st.error("Case not found!")

def app():
    st.title("Expert Review Dashboard")

    if 'transaction_analysis_completed' not in st.session_state or not st.session_state['transaction_analysis_completed']:
        st.error("Please complete the transaction analysis before proceeding to the expert review dashboard.")
        return

    if 'case_review_df' in st.session_state and st.session_state['case_review_df'] is not None:
        case_review_df = st.session_state['case_review_df']

        # Drop unnecessary columns for the display
        columns_to_drop = ['RF Approval Status', 'LOF Status', 'LOF Status_x', 'rf_predicted_fraud', 'LOF Status_y', 'lof_scores_y']
        case_review_df = case_review_df.drop(columns=columns_to_drop, errors='ignore')

        # Simulate offline review if not done already
        if 'offline_review_simulated' not in st.session_state:
            case_review_df = simulate_offline_review(case_review_df)
            st.session_state['case_review_df'] = case_review_df
            st.session_state['offline_review_simulated'] = True
            st.success("Offline review simulation complete. Expert decisions have been added.")

        col1, col2 = st.columns(2)
        with col1:
            plot_workflow_diagram(case_review_df)
        with col2:
            plot_case_resolution_timeline(case_review_df)

        case_id_option = st.selectbox("Select a case to review in detail:", case_review_df['ref_id'].unique())
        show_case_detail(case_review_df, case_id_option)

        st.subheader("Updated Transactions after Expert Review")
        st.dataframe(case_review_df)
       st.session_state['outcome_df'] = case_review_df
    else:
        st.error("No transaction data available for review. Please analyze transactions first.")

app()
