from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware
)

from pydantic import BaseModel

import pandas as pd
import joblib


# ---------------------------------------
# 1. Create FastAPI Application
# ---------------------------------------

app = FastAPI(

    title="House Price Prediction API",

    description="ML API for predicting house prices",

    version="1.0.0"
)


# ---------------------------------------
# 2. Enable CORS
# ---------------------------------------

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ---------------------------------------
# 3. Load Model
# ---------------------------------------

model = joblib.load(

    "models/house_price_model.pkl"

)


# ---------------------------------------
# 4. Input Schema
# ---------------------------------------

class HouseInput(BaseModel):

    location: str

    size: float

    rooms: int

    age: float


# ---------------------------------------
# 5. Home
# ---------------------------------------

@app.get("/")
def home():

    return {

        "message":
        "House Price Prediction API is running!"

    }


# ---------------------------------------
# 6. Health
# ---------------------------------------

@app.get("/health")
def health():

    return {

        "status": "healthy"

    }


# ---------------------------------------
# 7. Prediction
# ---------------------------------------

@app.post("/predict")
def predict_house_price(

    house: HouseInput

):

    input_data = pd.DataFrame([

        {

            "location":
                house.location,

            "size":
                house.size,

            "rooms":
                house.rooms,

            "age":
                house.age

        }

    ])


    prediction = model.predict(

        input_data

    )


    return {

        "location":
            house.location,

        "size":
            house.size,

        "rooms":
            house.rooms,

        "age":
            house.age,

        "predicted_price":
            float(prediction[0])

    }