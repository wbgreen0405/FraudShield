import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.chart_helpers import plot_confusion_matrix, plot_roc_curve

# Assuming `plot_feature_importances` is defined in `utils/chart_helpers`
from utils.chart_helpers import plot_feature_importances

def supervised_fraud_results_page():
    st.title('Supervised Fraud Model Results')
    
    if 'rf_model' in st.session_state and 'y_true' in st.session_state:
        # Retrieve the stored data and metrics
        rf_model = st.session_state['rf_model']
        y_true = st.session_state['y_true']
        y_pred = st.session_state['y_pred']
        y_prob = st.session_state['y_prob']
        model_metrics = st.session_state['model_metrics']

        # Display model metrics
        st.subheader('Model Metrics')
        st.json(model_metrics)

        # Generate and display feature importances chart if the model has the attribute
        st.subheader('Feature Importances')
        if hasattr(rf_model, 'feature_importances_'):
            feature_importances = rf_model.feature_importances_
            indices = np.argsort(feature_importances)[::-1]  # Sort the feature importances in descending order
            plt.figure()
            plt.title("Feature Importances")
            plt.bar(range(len(feature_importances)), feature_importances[indices], align="center")
            plt.xticks(range(len(feature_importances)), indices)
            plt.xlim([-1, len(feature_importances)])
            st.pyplot(plt)
        else:
            st.error('Model does not have feature importances.')

        # Generate and display confusion matrix
        st.subheader('Confusion Matrix')
        plot_confusion_matrix(y_true, y_pred)

        # Generate and display ROC curve
        st.subheader('ROC Curve')
        plot_roc_curve(y_true, y_prob)

    else:
        st.error('Model and prediction data are not available. Please run predictions first.')

# Run this page function
supervised_fraud_results_page()

