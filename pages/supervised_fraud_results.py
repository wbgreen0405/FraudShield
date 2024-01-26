import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.chart_helpers import plot_confusion_matrix, plot_roc_curve

# Import the plot_feature_importances function
from utils.chart_helpers import plot_feature_importances

def supervised_fraud_results_page():
    st.title('Supervised Fraud Model Results')
    
    if 'rf_model' in st.session_state and 'y_true' in st.session_state:
        rf_model = st.session_state['rf_model']
        y_true = st.session_state['y_true']
        y_pred = st.session_state['y_pred']
        y_prob = st.session_state['y_prob']
        model_metrics = st.session_state.get('model_metrics', {})

        # Display model metrics
        st.subheader('Model Metrics')
        st.json(model_metrics)

        # Generate and display feature importances chart if the model has the attribute
        st.subheader('Feature Importances')
        if hasattr(rf_model, 'feature_importances_'):
            plt = plot_feature_importances(rf_model, rf_model.feature_names_in_)
            st.pyplot(plt)
        else:
            st.error('Model does not have feature importances.')

        # Generate and display confusion matrix
        st.subheader('Confusion Matrix')
        plt = plot_confusion_matrix(y_true, y_pred)
        st.pyplot(plt)

        # Generate and display ROC curve
        st.subheader('ROC Curve')
        plt = plot_roc_curve(y_true, y_prob)
        st.pyplot(plt)

    else:
        st.error('Model and prediction data are not available. Please run predictions first.')

# Run this page function
supervised_fraud_results_page()


