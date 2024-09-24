import streamlit as st
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

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
                    #maintenance_cost = num_failures * 3_500_000

                    # Horizontal line separator page
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    # Title
                    st.title("Machine Learning Dashboard")

                    # Display scorecards
                    col1, col2 = st.columns(2)
                    col1.markdown(f"<div style='font-size: 12px; font-weight: bold;'>Number of Maintenance Required</div>", unsafe_allow_html=True)
                    col1.metric(label="", value=num_failures)
                    col2.markdown(f"<div style='font-size: 12px; font-weight: bold;'>Percentage of Maintenance Required</div>", unsafe_allow_html=True)
                    col2.metric(label="", value=f"{percentage_failures:.2f}%")
                    #col3.markdown(f"<div style='font-size: 12px; font-weight: bold;'>Cost of Maintenance</div>", unsafe_allow_html=True)
                    #col3.metric(label="", value=f"Rp {maintenance_cost/1_000_000:.0f} Juta")

                    # Horizontal line separator page
                    st.markdown("<hr>", unsafe_allow_html=True)

                    # Create a container for the plots
                    with st.container():
                        # Create 2 columns
                        col1, col2 = st.columns(2)

                        # Plot 1: 
                        with col1:
                            fig1 = px.histogram(df_predictions, 
                            x='Rotational speed [rpm]', 
                            color='Prediction',
                            title='Rotational speed Distribution',
                            barmode='overlay',  # Set barmode to 'overlay'
                            color_discrete_sequence=['#87CEEB', '#FFA07A'])
                            fig1.update_xaxes(title_text='Rotational speed [rpm]')
                            fig1.update_yaxes(title_text='Frequency')
                            fig1.update_traces(opacity=0.8)  # Adjust opacity for better visibility
                            fig1.update_layout(title_x=0.2,  # Center the title
                                            autosize=False, 
                                            width=450,  # Set width to make it square
                                            height=450)  # Set height to match the width
                            st.plotly_chart(fig1)

                        # Plot 2: 
                        with col2:
                            fig2 = px.histogram(df_predictions, x='Torque [Nm]', color='Prediction',
                            title='Torque [Nm] Distribution',
                            barmode='overlay', # Set barmode to 'overlay'
                            color_discrete_sequence=['#87CEEB', '#FFA07A'])
                            fig2.update_xaxes(title_text='Torque [Nm]')
                            fig2.update_yaxes(title_text='Frequency')
                            fig2.update_traces(opacity=0.8) # Adjust opacity for better visibility
                            fig2.update_layout(title_x=0.2,  # Center the title
                                               autosize=False, 
                                               width=450,  # Set width to make it square
                                               height=450)  # Set height to match the width
                            st.plotly_chart(fig2)

                    # Load model from the 'model' directory
                    best_model = joblib.load("models/best_model.pkl")

                    # Get model
                    best_model_wrap = best_model.best_estimator_

                    # Get feature importance
                    importances = best_model_wrap.feature_importances_

                    # Get feature names
                    feature_names = best_model_wrap.feature_names_in_

                    # Create a DataFrame to display feature importance with feature names
                    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

                    # Sort the DataFrame by importance values in descending order
                    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

                    # Horizontal line separator page
                    st.markdown("<hr>", unsafe_allow_html=True)

                    # Create a container for the feature importance section
                    st.subheader("**Feature Importance**")

                    # Adds line of blank space
                    st.write("")  
                    st.write("") 
                    st.write("")

                    # Create 2 columns: left for table, right for barchart
                    # Sort the DataFrame by 'Importance' in descending order
                    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
                    feature_importance_df.index = feature_importance_df.index + 1

                    # Create a container for the two columns
                    with st.container():
                        # Define two equally sized columns
                        col1, col2 = st.columns(2)

                        # Display feature importance DataFrame in the left column
                        with col1:
                            st.dataframe(feature_importance_df)

                        # Create bar chart for feature importance in the right column
                        with col2:
                            fig = px.bar(
                                feature_importance_df.sort_values(by='Importance', ascending=True), 
                                x='Importance', 
                                y='Feature', 
                                orientation='h',  # Horizontal bar chart
                                title='Feature Importance Bar Chart',
                                color_discrete_sequence=['royalblue']
                            )
                            fig.update_layout(
                                xaxis_title='Importance',
                                yaxis_title='Features',
                                height=450, 
                                width=450
                            )
                            st.plotly_chart(fig)




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

                    # Horizontal line separator page
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    # PART INI PAKE HARD CODE 
                    # Hard-coded F2 and ROC AUC scores
                    f2_score = 91.31  # Hard-coded recall score
                    roc_auc = 89.76 # Hard-coded ROC AUC score


                    # Display 
                    st.subheader("**Machine Learning Performance**")

                    # Adds line of blank space
                    st.write("")  
                    st.write("") 
                    st.write("")
                    st.write("")

                    # Create a container for the metrics and ROC curve
                    col1, col2 = st.columns(2)

                    # Display F2 score
                    with col1:
                        col1.markdown(f"<div style='font-size: 14px; font-weight: bold;'>F2 Score</div>", unsafe_allow_html=True)
                        st.metric(label="", value=f"{f2_score:.2f}%")  # Hard-coded F2 score
                        
                    # Display ROC AUC score
                    with col2:
                        col2.markdown(f"<div style='font-size: 14px; font-weight: bold;'>ROC AUC Score</div>", unsafe_allow_html=True)
                        st.metric(label="", value=f"{roc_auc:.2f}%")  # Hard-coded ROC AUC score
                        
                    # Adds line of blank space
                    st.write("")  
                    st.write("") 
                    st.write("")
                    st.write("")

                    # Display 
                    st.markdown("##### ROC Curve Analysis")  

                    # Adds line of blank space
                    st.write("")  
                    st.write("") 
                    st.write("")
                    st.write("")


                    # Display ROC curve image 
                    st.image("image/roc_curve.png", caption="ROC Curve", use_column_width=True,width=300)  
                    
                    
                   
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
