import streamlit as st
import pandas as pd
import requests
import json

def predict_dataset():
    st.title("Predictive Maintenance")
    st.write("Upload a CSV file containing the predictors and get predictions.")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file with predictors", type="csv")

    if uploaded_file is not None:
        # Read the uploaded file into a DataFrame
        data = pd.read_csv(uploaded_file)
        
        # Display the first few rows of the uploaded file
        st.write("**Data Preview**")
        st.dataframe(data.head())

        # Convert DataFrame to JSON format
        data_json = data.to_dict(orient="records")  # Convert to list of dicts
        
        # Add a predict button
        if st.button("Predict"):
            # Display processing message
            st.write("Processing data and making predictions...")

            try:
                # Debug print statement before sending request
                #st.write("Sending data to FastAPI endpoint for prediction...")
                print("Sending data to FastAPI endpoint for prediction...")
                print(f"Data being sent: {json.dumps(data_json, indent=2)}")

                # Make prediction request
                response = requests.post("http://localhost:8000/predict_dataset", json={"data": data_json})

                # Debug print statement after receiving response
                #st.write(f"Received response with status code: {response.status_code}")
                print(f"Received response with status code: {response.status_code}")
                print(f"Response content: {response.content.decode()}")

                # Check if the request was successful
                if response.status_code == 200:
                    # Get predictions from the response
                    predictions = response.json()
                    df_predictions = pd.DataFrame(predictions)
                    df_failure = df_predictions[df_predictions['Prediction'] == 1]

                    # Calculate number and percentage of failures
                    num_failures = df_predictions['Prediction'].sum()
                    percentage_failures = (num_failures / len(df_predictions)) * 100

                    # calculate cost
                    maintenance_cost = num_failures * 3_500_000

                    # Horizontal line separator page
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    # Title
                    st.title("Machine Learning Dashboard")

                    # Display scorecards
                    col1, col2, col3 = st.columns(3)
                    col1.markdown(f"<div style='font-size: 12px; font-weight: bold;'>Number of Maintenance Required</div>", unsafe_allow_html=True)
                    col1.metric(label="", value=num_failures)
                    col2.markdown(f"<div style='font-size: 12px; font-weight: bold;'>Percentage of Maintenance Required</div>", unsafe_allow_html=True)
                    col2.metric(label="", value=f"{percentage_failures:.2f}%")
                    col3.markdown(f"<div style='font-size: 12px; font-weight: bold;'>Cost of Maintenance</div>", unsafe_allow_html=True)
                    col3.metric(label="", value=f"Rp {maintenance_cost/1_000_000:.0f} Juta")

                    # Horizontal line separator page
                    st.markdown("<hr>", unsafe_allow_html=True)

                    # Display predictions with scroll capability
                    st.header("**Predicted Machine Failure Data**")
                    st.dataframe(df_failure, height=300)

                    # Download CSV and Send Email buttons side by side
                    col1_button, col2_button = st.columns(2)

                    with col1_button:
                        # Download CSV button
                        csv = df_failure.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Data Now",
                            data=csv,
                            file_name='data_machine_failure.csv',
                            mime='text/csv'
                        )

                    with col2_button:
                        # Send Me Email button
                        if st.button("Send to my Email"):
                            st.write("Email sent!", key="email_button_1")  
                    
                    # show all data
                    st.header("**All Predicted Result**")
                    st.dataframe(df_predictions, height=300) 
                    
                    # Download CSV and Send Email buttons side by side
                    col3_button, col4_button = st.columns(2)

                    with col3_button:
                        # Download CSV button
                        csv2 = df_predictions.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Data Now",
                            data=csv2,
                            file_name='data_machine.csv',
                            mime='text/csv'
                        )

                    with col4_button:
                        # Send Me Email button
                        if st.button("Send to my Email", key="email_button_2"):
                            st.write("Email sent!") 
                else:
                    st.error(f"Error occurred while processing the data. Status code: {response.status_code}")
                    print(f"Error occurred while processing the data. Status code: {response.status_code}")
                    print(f"Error details: {response.text}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                print(f"An error occurred: {str(e)}")

# Run the Streamlit app
if __name__ == "__main__":
    predict_dataset()
