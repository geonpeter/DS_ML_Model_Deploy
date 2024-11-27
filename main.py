import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fastapi import FastAPI
from mangum import Mangum
from preprocessing import preprocess_data,preprocess_newdata
from model_manager import predict_new,train_models
from datetime import datetime,date,time
from fastapi import HTTPException
import joblib
from config import DATA_PATH
from pydantic import BaseModel
import requests

app = FastAPI()
handler = Mangum(app)

class InputData(BaseModel):
    operator_code: str
    date: date
    time: time
    chosen_model: str

# API Endpoint for prediction
@app.post("/prediction")
def get_prediction(data:InputData):
    try:
        input_data = preprocess_newdata(data.operator_code, data.date, data.time)
        prediction = predict_new(data.chosen_model, input_data)
        if prediction.get("value") == 1:
            prediction["value"] = "Employee will be present"
        else:
            prediction["value"] = "Employee will not be present"
        return prediction   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
# API Endpoint for Analysis
@app.get("/analysis")
def get_analysis():
    try:
        X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH)
        results = train_models(X_train, X_test, y_train, y_test)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


    
