from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Iris Classification API", description="API for Iris flower species prediction")

# Define input schema
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Iris Classification API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Convert input to model format and scale
        features = np.array([[input_data.sepal_length, input_data.sepal_width, 
                            input_data.petal_length, input_data.petal_width]])
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        # Get prediction probabilities
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities))
        
        return PredictionOutput(prediction=prediction, confidence=confidence)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "problem_type": "classification",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    }