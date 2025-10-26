"""
Sentinel-API Main Application
FastAPI application for ICU mortality prediction

Author: Sanwal Ahmad Zafar
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Sentinel-API", description="ICU Mortality Prediction API", version="1.0.0")

class PredictionRequest(BaseModel):
    age: float
    gender: str
    hr_mean: Optional[float] = None
    sbp_mean: Optional[float] = None

class PredictionResponse(BaseModel):
    probability: float
    risk_category: str
    confidence: float

@app.get('/')
def root():
    return {'message': 'Sentinel-API is running', 'version': '1.0.0'}

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return PredictionResponse(probability=0.0, risk_category='low', confidence=0.95)

@app.get('/health')
def health_check():
    return {'status': 'healthy'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
