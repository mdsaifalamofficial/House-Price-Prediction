# House Price Prediction

A complete Machine Learning web application that predicts an estimated house price based on location, house size, number of rooms, and house age.

## Features

- Data cleaning
- Exploratory Data Analysis
- Feature engineering
- Linear Regression
- Random Forest Regression
- Model evaluation
- Machine Learning pipeline
- Saved model
- FastAPI backend
- HTML/CSS/JavaScript frontend
- End-to-end prediction

## Input Features

- Location
- Size
- Rooms
- Age

## Target

- House Price

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- HTML
- CSS
- JavaScript

## Installation

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

## Run Backend

uvicorn api.main:app --reload

API:

http://127.0.0.1:8000

Swagger:

http://127.0.0.1:8000/docs

## Run Frontend

Open:

frontend/index.html

using VS Code Live Server.

## API Endpoint

POST /predict

Example request:

{
    "location": "Patna",
    "size": 1800,
    "rooms": 3,
    "age": 5
}

## Important Limitation

This project uses a very small 20-row dataset and is intended for educational purposes. Its predictions should not be treated as real-world property valuations.