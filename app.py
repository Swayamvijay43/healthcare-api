from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

heart_model = xgb.XGBClassifier()
heart_model.load_model("heart_model.json")

liver_model = xgb.XGBClassifier()
liver_model.load_model("liver_model.json")


class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class LiverInput(BaseModel):
    Age: float
    Gender: float
    Total_Bilirubin: float
    Direct_Bilirubin: float
    Alkaline_Phosphotase: float
    Alamine_Aminotransferase: float
    Aspartate_Aminotransferase: float
    Total_Protiens: float
    Albumin: float
    Albumin_and_Globulin_Ratio: float


@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    df = pd.DataFrame([data.dict()])
    prob = float(heart_model.predict_proba(df)[0][1])
    return {
        "probability": round(prob * 100, 1),
        "risk": "HIGH RISK" if prob > 0.5 else "LOW RISK"
    }


@app.post("/predict/liver")
def predict_liver(data: LiverInput):
    df = pd.DataFrame([data.dict()])
    prob = float(liver_model.predict_proba(df)[0][1])
    return {
        "probability": round(prob * 100, 1),
        "risk": "HIGH RISK" if prob > 0.5 else "LOW RISK"
    }
