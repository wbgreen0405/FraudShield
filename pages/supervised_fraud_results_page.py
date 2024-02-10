import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from supabase import create_client, Client

# Initialize Supabase client using Streamlit secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase: Client = create_client(supabase_url, supabase_key)

def supervised_fraud_results_page():
    st.title('Supervised Fraud Model Results')

    # Fetch model results from Supabase tables
    rf_confusion_matrix = fetch_supabase_table("rf_confusion_matrix")
    rf_feature_importance = fetch_supabase_table("rf_feature_importances")
    rf_model_metrics = fetch_supabase_table("rf_model_metrics")

    # Display the fetched results
    st.subheader("Random Forest Confusion Matrix:")
    plot_confusion_matrix(rf_confusion_matrix)

    st.subheader("Random Forest Feature Importance:")
    plot_feature_importance(rf_feature_importance)

    st.subheader("Random Forest Model Metrics:")
    st.dataframe(rf_model_metrics)

def plot_confusion_matrix(df):
    if not df.empty:
        # Assuming your confusion matrix is in a DataFrame with appropriate labels
        fig = ff.create_annotated_heatmap(
            z=df.values,
            x=list(df.columns),
            y=list(df.index),
            annotation_text=df.values,
            colorscale='Viridis'
        )
        fig.update_layout(title_text='Confusion Matrix', xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig)
    else:
        st.write("Confusion matrix data not available.")

def plot_feature_importance(df):
    if not df.empty:
        # Assuming your feature importance DataFrame has 'Feature' and 'Importance' columns
        fig = px.bar(df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
        st.plotly_chart(fig)
    else:
        st.write("Feature importance data not available.")

# Helper function to fetch data from Supabase tables
def fetch_supabase_table(table_name):
    try:
        response = supabase.table(table_name).select('*').execute()
        # Directly create a DataFrame from the response if it's successful
        if response:
            return pd.DataFrame(response)
        else:
            st.error(f'Unexpected response format or empty data from {table_name}.')
            return pd.DataFrame()
    except Exception as e:
        st.error(f'An error occurred while fetching data from {table_name}: {e}')
        return pd.DataFrame()


if __name__ == '__main__':
    supervised_fraud_results_page()
