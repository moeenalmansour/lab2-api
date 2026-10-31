from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

model = None

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)


class CustomerInput(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    support_calls: int = Field(ge=0)
    contract: str = Field(min_length=1)
    internet_service: str = Field(min_length=1)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(customer: CustomerInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    data = pd.DataFrame([customer.model_dump()])

    probability = float(
        model.predict_proba(data)[0][1]
    )

    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "result": "Churn" if prediction == 1 else "No Churn"
    }
