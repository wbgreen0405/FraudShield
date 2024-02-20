import streamlit as st
from PIL import Image

def app():
    st.title('Welcome to the Fraud Detection System Demo.')

    # Use columns to layout image and text
    col1, col2 = st.columns([1, 2])

    # Display the pipeline image in the left column
    with col1:
        pipeline_image_path = 'assets/Fraud Detection.png'  # Correct path to the image file
        pipeline_image = Image.open(pipeline_image_path)
        # Set width to 300 pixels (or any other size you wish)
        st.image(pipeline_image, caption="Fraud Detection Pipeline Visualization", width=300)

    # Put the text content in the right column
    with col2:
        st.markdown("""
        Dive into the world of anti-fraud analytics with our streamlined, user-friendly platform designed to highlight suspicious financial activities through data-driven insights. Our system provides a comprehensive suite of tools to identify, analyze, and combat fraudulent transactions, ensuring your financial security.

        **Inspired by "Learning Fraud Anomalies, a Semi-Supervised Mouse Trap!" by Scienaptic:**
        
        Imagine a blind person leading one with good vision through a complex maze. This counter-intuitive scenario is akin to building robust fraud engines using semi-supervised approaches. Fraud is evolving, with new patterns constantly emerging, making it challenging to train supervised models. By stacking an unsupervised anomaly detector atop the trained fraud model, we maintain a "state" information about transactions, enabling us to discern between genuine activities and potential fraud.

        **Key Features:**
        - Secure data upload and preprocessing.
        - Pre-trained Random Forest classifier for identifying potential fraud.
        - Local Outlier Factor model for scrutinizing flagged transactions.
        - Simulation of expert review decisions to validate flagged cases.
        - Dataset update based on review outcomes for further analysis or model retraining.

        **How to Use:** Navigate through the sidebar to access different stages of the fraud detection process. Start by uploading your dataset and then proceed through each stage to explore the full capabilities of the system.

        Our app offers an in-depth look at how data science can be leveraged to protect financial assets and maintain transactional integrity.

        **Dataset Information:**
        The dataset used in this demo is based on the "Bank Account Fraud Dataset" from NeurIPS 2022, provided by Scienaptic. For more details about the dataset and the methodologies used, you can access the [dataset on Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) and refer to the publication:

        > Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R. P., Gama, J., & Bizarro, P. (2022). Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation. Advances in Neural Information Processing Systems.
        
        **Citation:**
        ```
        @article{jesusTurningTablesBiased2022,
          title={Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation},
          author={Jesus, Sérgio and Pombal, José and Alves, Duarte and Cruz, André and Saleiro, Pedro and Ribeiro, Rita P. and Gama, João and Bizarro, Pedro},
          journal={Advances in Neural Information Processing Systems},
          year={2022}
        }
        ```
        """)

# Run the app function
app()


