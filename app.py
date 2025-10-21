from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

class FuelInput(BaseModel):
    engine_size: float
    cylinders: int

app = FastAPI(title="Fuel Consumption Prediction API", version="1.0.0")

# Load model with better error handling
try:
    model_path = os.path.join('models', 'fuel_consumption_model.pkl')
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
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