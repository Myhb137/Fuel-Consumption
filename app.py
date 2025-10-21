from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

class FuelInput(BaseModel):
    engine_size: float
    cylinders: int

app = FastAPI()

try:
    model = joblib.load('models/fuel_consumption_model.pkl')
except:
    model = None

@app.get("/")
def home():
    return {"message": "Fuel Consumption API is running"}

@app.post("/predict")
def predict(data: FuelInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if data.engine_size <= 0 or data.cylinders <= 0:
        raise HTTPException(status_code=400, detail="Invalid input values")
    
    input_values = [[data.engine_size, data.cylinders]]
    prediction = model.predict(input_values)
    
    return {"prediction": float(prediction[0])}