import streamlit as st
import pandas as pd
import plotly.express as px
import random

def simulate_offline_review(review_df):
    for index, row in review_df.iterrows():
        review_df.at[index, 'expert_decision'] = random.choice(['Confirmed Fraud', 'Confirmed Legitimate'])
    return review_df

def plot_workflow_diagram(review_df):
    fig = px.pie(review_df, names='expert_decision', title='Review Cases Status')
    st.plotly_chart(fig)

def plot_case_resolution_timeline(review_df):
    if all(col in review_df.columns for col in ['review_start', 'review_end', 'ref_id']):
        review_df['review_start'] = pd.to_datetime(review_df['review_start'], errors='coerce')
        review_df['review_end'] = pd.to_datetime(review_df['review_end'], errors='coerce')
        fig = px.timeline(review_df, x_start='review_start', x_end='review_end', y='ref_id', labels={'ref_id': 'Case ID'})
        fig.update_layout(xaxis_title='Time', yaxis_title='Case ID', title='Case Resolution Timeline')
        fig.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig)
    else:
        st.error("Dataframe does not contain the required columns for plotting the timeline.")

def show_case_detail(review_df, case_id):
    case_data = review_df[review_df['ref_id'] == case_id]
    if not case_data.empty:
        st.write(case_data)
    else:
        st.error("Case not found!")

def app():
    st.title("Expert Review Dashboard")
    if 'review_df' in st.session_state:
        review_df = st.session_state['review_df']

        if st.button('Simulate Offline Review'):
            review_df = simulate_offline_review(review_df)
            st.session_state['review_df'] = review_df
            st.success("Simulation complete. Expert decisions have been added.")

        plot_workflow_diagram(review_df)
        plot_case_resolution_timeline(review_df)
        
        case_id_option = st.selectbox("Select a case to review in detail:", review_df['ref_id'].unique())
        show_case_detail(review_df, case_id_option)

        st.subheader("Updated Transactions after Expert Review")
        st.dataframe(review_df)
    else:
        st.error("No transaction data available for review. Please analyze transactions first.")

if __name__ == '__main__':
    app()
