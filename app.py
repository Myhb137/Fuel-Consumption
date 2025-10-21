from fastapi import FastAPI
from pydantic import BaseModel
import joblib 


class inputs(BaseModel):
    input1: float
    input2: float
    
model = joblib.load('models/fuel_consumption_model.pkl')

app = FastAPI() 

@app.post('/predict/')
def predict(data : inputs):
    input_values = [[data.input1, data.input2]]
    
    
    prediction = model.predict(input_values)
    return {'prediction': prediction[0]}