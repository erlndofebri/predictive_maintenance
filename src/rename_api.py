# Import Library
import pandas as pd
import utils as utils
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from data_preprocessing import preprocess_data

# Buat fungsi & class
class ApiData(BaseModel):
    types : str
    air_temperature : float
    process_temperature : float
    rotational_speed : int
    torque : float
    tool_wear : int

# load config data
CONFIG_DATA = utils.config_load()

# Buat Apps
app = FastAPI()

# Buat dekorator alamat home
@app.get('/')
def home():
    return {'Text': 'Alamat Home'}


# Buat dekorator alamat prediksi
@app.post('/predict')
def predict(data: ApiData):
    # Convert data api to dataframe
    data_dict = data.dict()
    data_dict = {key:[val] for key, val in data_dict.items()}

    data = pd.DataFrame(data_dict)
    data.columns = ['Type', 'Air temperature [K]', 'Process temperature [K]',
                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # Lakukan preprocess data
    data_clean = preprocess_data(X=data, types=None, CONFIG_DATA=CONFIG_DATA)

    # Prediksi data
    # Load model
    model = utils.pickle_load(CONFIG_DATA['best_model_path'])
    y_pred_proba = model.predict_proba(data_clean)[:, 1]

    # Load threshold
    thresh = utils.pickle_load(CONFIG_DATA['best_threshold_path'])
    y_pred = (y_pred_proba >= thresh).astype(int)

    # Return
    return {
        "res": int(y_pred[0]),
        "res_proba": float(y_pred_proba[0]),
        "error_msg": ""
    }


# Panggil program
if __name__ == '__main__':
    uvicorn.run('api:app',
                host = '127.0.0.1',
                port = 8000)