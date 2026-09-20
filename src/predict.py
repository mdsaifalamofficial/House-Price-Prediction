import pandas as pd
import joblib


# -----------------------------------
# 1. Load Trained Pipeline
# -----------------------------------

pipeline = joblib.load(
    "models/house_price_model.pkl"
)

print("Trained pipeline loaded successfully!")


# -----------------------------------
# 2. User Input
# -----------------------------------

new_house = pd.DataFrame({

    "location": ["Delhi"],

    "size": [1600],

    "rooms": [3],

    "age": [5]

})


print("\nHouse Details:")
print(new_house)


# -----------------------------------
# 3. Prediction
# -----------------------------------

predicted_price = pipeline.predict(
    new_house
)


# -----------------------------------
# 4. Display Result
# -----------------------------------

print("\n==============================")
print("HOUSE PRICE PREDICTION")
print("==============================")

print(
    f"Estimated House Price: ₹{predicted_price[0]:,.2f}"
)