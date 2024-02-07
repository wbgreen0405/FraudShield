import streamlit as st
import pandas as pd
import random
import plotly.express as px

# Placeholder for the review dataframe
if 'review_df' in st.session_state:
        # Retrieve the analyzed dataframe from the session state
        review_df = st.session_state['review_df']
        
def simulate_offline_review(transaction_data):
    # Constants defining the fraud criteria
    INCOME_THRESHOLD = 100000
    AGE_THRESHOLD = 50
    EMPLOYMENT_STATUS_SUSPICIOUS = 3
    HOUSING_STATUS_SUSPICIOUS = 2
    ERROR_RATE = 0.1

    decisions = {}
    for index, transaction in transaction_data.iterrows():
        # Apply expert rules to decide if a transaction is fraudulent
        is_unusually_high_income = transaction['income'] > INCOME_THRESHOLD
        is_age_above_threshold = transaction['customer_age'] > AGE_THRESHOLD
        is_suspicious_employment = transaction['employment_status'] == EMPLOYMENT_STATUS_SUSPICIOUS
        is_suspicious_housing = transaction['housing_status'] == HOUSING_STATUS_SUSPICIOUS

        decision = 'fraudulent' if is_unusually_high_income and (is_age_above_threshold or is_suspicious_employment or is_suspicious_housing) else 'legitimate'

        # Introduce a random error rate to simulate inaccuracies in expert decision-making
        if random.random() < ERROR_RATE:
            decision = 'legitimate' if decision == 'fraudulent' else 'fraudulent'

        decisions[index] = decision
    return decisions

def show_case_detail(case_id, data):
    st.header(f"Case Detail: {case_id}")
    case = data[data['ref_id'] == case_id]
    if not case.empty:
        st.write(case)
    else:
        st.error("Case not found!")

def plot_workflow_diagram(data):
    st.header("Workflow Diagram")
    fig = px.pie(data, names='expert_decision', title='Review Cases Status')
    st.plotly_chart(fig)

def plot_case_resolution_timeline(data):
    st.header("Case Resolution Timeline")
    fig = px.timeline(data, x_start='review_start', x_end='review_end', y='ref_id', labels={'ref_id': 'Case ID'})
    fig.update_yaxes(autorange="reversed")  # To reverse the Y-Axis
    st.plotly_chart(fig)

# Streamlit app
def app():
    st.title("Expert Review Dashboard")

    # Workflow Diagram
    plot_workflow_diagram(review_df)

    # Case Resolution Timeline
    plot_case_resolution_timeline(review_df)

    # Display transactions
    st.header("Transactions")
    st.dataframe(review_df)

    # Simulate button
    if st.button('Simulate Offline Review'):
        decisions = simulate_offline_review(review_df)
        review_df['expert_decision'] = review_df.index.map(decisions)
        st.write("Simulation complete. Expert decisions have been added.")

    # Individual Case Detail Page
    case_id = st.selectbox("Select a case to review in detail:", review_df['ref_id'])
    show_case_detail(case_id, review_df)

    # Display updated dataframe
    st.header("Updated Transactions after Expert Review")
    st.dataframe(review_df)

    # Optional: Save the updated dataframe to a CSV file
    if st.button('Save Updated Data'):
        review_df.to_csv('updated_transactions.csv', index=False)
        st.success('Data saved successfully!')

app()
