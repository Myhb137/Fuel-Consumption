from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os

# -------------------------------------------------
# Model setup
# -------------------------------------------------
class FuelInput(BaseModel):
    engine_size: float
    cylinders: int

app = FastAPI(title="Fuel Consumption Prediction API", version="1.0.0")

# Load model with error handling
try:
    model_path = os.path.join('models', 'fuel_consumption_model.pkl')
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# -------------------------------------------------
# Serve your HTML frontend
# -------------------------------------------------
# Assuming your HTML file is named 'index.html' and is in a folder named 'static'
# (You can also just put it in the same folder as this script and adjust the path below)
@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return HTMLResponse("<h1>Fuel Consumption API</h1><p>index.html not found.</p>", status_code=404)


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
async def predict(data: FuelInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if data.engine_size <= 0 or data.cylinders <= 0:
        raise HTTPException(status_code=400, detail="Invalid input values")

    input_values = [[data.engine_size, data.cylinders]]
    prediction = model.predict(input_values)
    return JSONResponse({"prediction": float(prediction[0])})
