import streamlit as st
from PIL import Image

def app():
    st.title('Welcome to the Fraud Detection System Demo.')

    # Display the pipeline image
    pipeline_image_path = 'assets/Fraud Detection.png'  # Path to the image file
    pipeline_image = Image.open(pipeline_image_path)
    st.image(pipeline_image, caption="Fraud Detection Pipeline Visualization")

    # Rest of the home page content with the article text
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
      author={Jesus, S{\'e}rgio and Pombal, Jos{\'e} and Alves, Duarte and Cruz, Andr{\'e} and Saleiro, Pedro and Ribeiro, Rita P. and Gama, Jo{\~a}o and Bizarro, Pedro},
      journal={Advances in Neural Information Processing Systems},
      year={2022}
    }
    ```
    """)


# Run the app function
app()

