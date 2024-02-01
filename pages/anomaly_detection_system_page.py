# anomaly_detection_system_page.py

import streamlit as st

def anomaly_detection_system_page():
    st.title('Anomaly Detection System')

    # Fetch the anomaly detection results
    anomaly_detection_results = st.session_state.get('anomaly_detection_results', None)

    if anomaly_detection_results is not None:
        st.subheader('Anomaly Detection Results')
        st.dataframe(anomaly_detection_results)  # Assuming it's a DataFrame

        # Provide details on the anomaly score and threshold settings
        st.subheader('Anomaly Score Details')
        for index, anomaly in anomaly_detection_results.iterrows():
            st.markdown(f"**Transaction ID:** {anomaly['ref_id']}")
            st.markdown(f"**Anomaly Score:** {anomaly['anomaly_score']:.2f}")
            st.markdown(f"**Threshold:** {anomaly['threshold']:.2f}")
            if anomaly['anomaly_score'] > anomaly['threshold']:
                st.error('This transaction is flagged as a potential anomaly.')
            else:
                st.success('This transaction is not considered an anomaly.')
    else:
        st.write("No results are available from the anomaly detection system at the moment.")

if __name__ == '__main__':
    anomaly_detection_system_page()
