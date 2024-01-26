import streamlit as st
import matplotlib.pyplot as plt
from utils.chart_helpers import plot_confusion_matrix, plot_roc_curve, plot_feature_importances

def supervised_fraud_results_page():
    st.title('Supervised Fraud Model Results')
    
    if 'y_true' in st.session_state and 'y_pred' in st.session_state and 'y_prob' in st.session_state:
        # Retrieve the stored data and metrics
        y_true = st.session_state['y_true']
        y_pred = st.session_state['y_pred']
        y_prob = st.session_state['y_prob']
        model_metrics = st.session_state['model_metrics']

        # Display model metrics
        st.subheader('Model Metrics')
        st.json(model_metrics)

        # Generate and display charts
        st.subheader('Feature Importances')
        # You need to have the trained model object available here to plot feature importances
        # model = st.session_state['model']
        # plot_feature_importances(model)

        st.subheader('Confusion Matrix')
        plot_confusion_matrix(y_true, y_pred)

        st.subheader('ROC Curve')
        plot_roc_curve(y_true, y_prob)

    else:
        st.error('No prediction data available. Please run predictions first.')

# Run this page function
supervised_fraud_results_page()
