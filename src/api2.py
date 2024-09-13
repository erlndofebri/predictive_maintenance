import pandas as pd
import utils as utils
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict
from data_preprocessing import preprocess_data

# Define classes for single data and batch data
class ApiData(BaseModel):
    types: str
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int

class Dataset(BaseModel):
    data: List  # Adjusted to use Dict[str, float]

# Load configuration data
CONFIG_DATA = utils.config_load()

# Create FastAPI app
app = FastAPI()

# Home endpoint
@app.get('/')
def home():
    return {'Text': 'Welcome to the FastAPI server!'}

# Endpoint for predicting a single data entry
@app.post('/predict')
def predict(data: ApiData):
    try:
        # Convert data to DataFrame
        data_dict = data.dict()
        data_dict = {key: [val] for key, val in data_dict.items()}
        data_df = pd.DataFrame(data_dict)
        data_df.columns = ['Type', 'Air temperature [K]', 'Process temperature [K]',
                           'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

        # Preprocess data
        data_clean = preprocess_data(X=data_df, types=None, CONFIG_DATA=CONFIG_DATA)

        # Load model and make predictions
        model = utils.pickle_load(CONFIG_DATA['best_model_path'])
        y_pred_proba = model.predict_proba(data_clean)[:, 1]
        thresh = utils.pickle_load(CONFIG_DATA['best_threshold_path'])
        y_pred = (y_pred_proba >= thresh).astype(int)

        # Return results
        return {
            "res": int(y_pred[0]),
            "res_proba": float(y_pred_proba[0]),
            "error_msg": ""
        }
    except Exception as e:
        return {
            "error_msg": str(e)
        }

# Endpoint for predicting a batch of data
@app.post("/predict_dataset")
def predict_dataset(dataset: Dataset):
    try:
        # Print the received data for debugging
        print("Received data:", dataset)

        # Convert the incoming JSON list to a DataFrame
        data_df = pd.DataFrame(dataset.data)
        data_df.columns = ['Type', 'Air temperature [K]', 'Process temperature [K]',
                           'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

        # Preprocess data
        data_clean = preprocess_data(X=data_df, types=None, CONFIG_DATA=CONFIG_DATA)

        # Load model
        model = utils.pickle_load(CONFIG_DATA['best_model_path'])

        # Make predictions
        y_pred_proba = model.predict_proba(data_clean)[:, 1]
        thresh = utils.pickle_load(CONFIG_DATA['best_threshold_path'])
        y_pred = (y_pred_proba >= thresh).astype(int)

        # Append predictions to DataFrame
        data_df['Prediction'] = y_pred
        data_df['Probability'] = y_pred_proba

        # Convert DataFrame to JSON for response
        return data_df.to_dict(orient='records')
    except Exception as e:
        print(f"Error: {str(e)}")  # Print error for debugging
        return {
            "error_msg": str(e)
        }

# Endpoint for handling raw file upload
#@app.post("/uploadfile/")
#async def upload_file(file: UploadFile = File(...)):
#    print("Diterima")
    # try:
    #     # Read file as a DataFrame
    #     df = pd.read_csv(file.file)
        
    #     # Display first few rows for debugging
    #     print("File content preview:")
    #     #print(df.head())

    #     # Convert DataFrame to JSON format
    #     data_json = df.to_dict(orient="records")
        
    #     # Preprocess data
    #     data_clean = preprocess_data(X=df, types=None, CONFIG_DATA=CONFIG_DATA)
        
    #     # Load model
    #     model = utils.pickle_load(CONFIG_DATA['best_model_path'])
        
    #     # Make predictions
    #     y_pred_proba = model.predict_proba(data_clean)[:, 1]
    #     thresh = utils.pickle_load(CONFIG_DATA['best_threshold_path'])
    #     y_pred = (y_pred_proba >= thresh).astype(int)
        
    #     # Append predictions to DataFrame
    #     df['Prediction'] = y_pred
    #     df['Probability'] = y_pred_proba

    #     # Convert DataFrame to JSON for response
    #     return df.to_dict(orient='records')
    # except Exception as e:
    #     print(f"Error: {str(e)}")  # Print error for debugging
    #     return {
    #         "error_msg": str(e)
    #     }

# Run the FastAPI server
if __name__ == '__main__':
    uvicorn.run('api2:app', host='127.0.0.1', port=8000)
