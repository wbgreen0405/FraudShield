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
    plot_model_metrics(rf_model_metrics)

def plot_confusion_matrix(df):
    if not df.empty:
        # Convert dataframe to matrix
        matrix = df.values
        fig = ff.create_annotated_heatmap(
            z=matrix,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            annotation_text=matrix.astype(str),
            colorscale='Viridis'
        )
        # Add title
        fig.update_layout(title_text="<b>Confusion Matrix</b>")
        # Add labels
        fig.update_xaxes(side="top")
        fig.update_yaxes(autorange="reversed")  # this is to show the actual negative at the top
        st.plotly_chart(fig)
    else:
        st.write("Confusion matrix data not available.")

def plot_feature_importance(df):
    if not df.empty:
        # Sort the DataFrame based on the 'Importance' column in descending order
        df_sorted = df.sort_values(by='Importance', ascending=True)
        
        # Assuming your feature importance DataFrame has 'Feature' and 'Importance' columns
        fig = px.bar(df_sorted, x='Importance', y='Feature', orientation='h', title="Feature Importance")
        st.plotly_chart(fig)
    else:
        st.write("Feature importance data not available.")

def plot_model_metrics(df):
    if not df.empty:
        # Drop the 'metric_id' column if it exists
        if 'metric_id' in df.columns:
            df = df.drop(columns=['metric_id'])

        # Reset the index of the DataFrame
        df = df.reset_index(drop=True)
        
        st.dataframe(df)
    else:
        st.write("Model metrics data not available.")

# Helper function to fetch data from Supabase tables
def fetch_supabase_table(table_name):
    try:
        response = supabase.table(table_name).select('*').execute()
        if hasattr(response, 'error') and response.error:
            st.error(f'Failed to retrieve data from {table_name}. Error: {str(response.error)}')
            return pd.DataFrame()
        elif hasattr(response, 'data'):
            return pd.DataFrame(response.data)
        else:
            st.error(f'Unexpected response format from {table_name}.')
            return pd.DataFrame()
    except Exception as e:
        st.error(f'An error occurred while fetching data from {table_name}: {e}')
        return pd.DataFrame()


if __name__ == '__main__':
    supervised_fraud_results_page()

