import streamlit as st
import pandas as pd
import plotly.express as px
import random
from datetime import datetime, timedelta

def apply_fraud_detection_rules(df):
    # Initialize a column for flagging potentially fraudulent transactions
    df['flagged_fraud'] = False

    # Rule 1: Low income and high proposed credit limit
    df.loc[(df['income'] < 0.3) & (df['proposed_credit_limit'] > 1500), 'flagged_fraud'] = True

    # Rule 2: Low similarity between name and email
    df.loc[df['name_email_similarity'] < 0.2, 'flagged_fraud'] = True

    # Rule 3: Short durations at both previous and current addresses
    df.loc[(df['prev_address_months_count'] < 6) & (df['current_address_months_count'] < 6), 'flagged_fraud'] = True

    # Rule 4: High velocity of applications in the last 24 hours or 4 weeks
    df.loc[(df['velocity_24h'] > 5000) | (df['velocity_4w'] > 5000), 'flagged_fraud'] = True

    # Rule 5: Negative credit risk score
    df.loc[df['credit_risk_score'] < 0, 'flagged_fraud'] = True

    # Rule 6: Use of free email domain and invalid mobile phone
    df.loc[(df['email_is_free'] == True) & (df['phone_mobile_valid'] == False), 'flagged_fraud'] = True

    # Rule 7: Foreign request with specific device OS
    df.loc[(df['foreign_request'] == True) & (df['device_os'] == 'other'), 'flagged_fraud'] = True

    # Rule 8: Previous fraudulent activity associated with the device
    df.loc[df['device_fraud_count'] > 0, 'flagged_fraud'] = True

    # Additional fraud detection rules can be added here following the same pattern:
    # df.loc[<condition>, 'flagged_fraud'] = True

    return df



def simulate_offline_review(review_df):
    if review_df is None:
        return None

    # Simulate review dates if not already present
    if 'review_start' not in review_df.columns or 'review_end' not in review_df.columns:
        review_start_dates, review_end_dates = [], []
        for _ in range(len(review_df)):
            start_date = datetime.now() - timedelta(days=random.randint(1, 30))
            end_date = start_date + timedelta(hours=random.randint(1, 48))
            review_start_dates.append(start_date)
            review_end_dates.append(end_date)
        review_df['review_start'] = review_start_dates
        review_df['review_end'] = review_end_dates

    # Simulate expert decisions considering flagged_fraud
    if 'expert_decision' not in review_df.columns:
        # Adjust decision-making to account for misclassification
        review_df['expert_decision'] = review_df['flagged_fraud'].apply(
            lambda flagged: 'Confirmed Fraud' if flagged else 'Confirmed Legitimate'
        )

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
    
    if 'transaction_analysis_completed' not in st.session_state or not st.session_state['transaction_analysis_completed']:
        st.error("Please complete the transaction analysis before proceeding to the expert review dashboard.")
        return
    
    if 'review_df' in st.session_state and st.session_state['review_df'] is not None:
        review_df = st.session_state['review_df']


        # Drop unnecessary columns
        columns_to_drop = ['RF Approval Status', 'LOF Status', 'LOF Status_x', 'rf_predicted_fraud', 'LOF Status_y', 'lof_scores_y']
        review_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

        # Apply fraud detection rules
        review_df = apply_fraud_detection_rules(review_df)

        # Simulate offline review considering flagged_fraud
        if 'offline_review_simulated' not in st.session_state:
            review_df = simulate_offline_review(review_df)
            st.session_state['review_df'] = review_df
            st.session_state['offline_review_simulated'] = True
            st.success("Offline review simulation complete. Expert decisions have been added.")

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

app()

