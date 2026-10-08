from fastapi import FastAPI
from pydantic import BaseModel
import joblib


# Create FastAPI application
app = FastAPI(
    title="Used Car Price Prediction API"
)


# Load trained model
model = joblib.load("model/used_car_model.pkl")


# Input structure
class CarData(BaseModel):

    car_age: float
    km_driven: float
    engine_cc: float
    previous_owners: float


# Home endpoint
@app.get("/")
def home():

    return {
        "message": "Used Car Price Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(car: CarData):

    input_data = [[
        car.car_age,
        car.km_driven,
        car.engine_cc,
        car.previous_owners
    ]]

    prediction = model.predict(input_data)

    return {
        "predicted_price_lakhs": round(
            float(prediction[0]), 2
        )
    }