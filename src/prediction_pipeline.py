import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------
# 1. Load Clean Dataset
# ---------------------------------------

df = pd.read_csv(
    "data/processed/house_data_clean.csv"
)

print("Clean dataset loaded successfully!")


# ---------------------------------------
# 2. Separate Features and Target
# ---------------------------------------

X = df[
    [
        "location",
        "size",
        "rooms",
        "age"
    ]
]

y = df["house_price"]


print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# ---------------------------------------
# 3. Define Feature Types
# ---------------------------------------

categorical_features = [
    "location"
]

numerical_features = [
    "size",
    "rooms",
    "age"
]


# ---------------------------------------
# 4. Create Preprocessor
# ---------------------------------------

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ---------------------------------------
# 5. Create Model
# ---------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ---------------------------------------
# 6. Create Pipeline
# ---------------------------------------

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ]
)


# ---------------------------------------
# 7. Train/Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)


print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)


# ---------------------------------------
# 8. Train Pipeline
# ---------------------------------------

pipeline.fit(
    X_train,
    y_train
)

print("\nPipeline training completed!")


# ---------------------------------------
# 9. Test Prediction
# ---------------------------------------

predictions = pipeline.predict(
    X_test
)

print("\nTest Predictions:")

print(predictions)


# ---------------------------------------
# 10. Create New House
# ---------------------------------------

new_house = pd.DataFrame({

    "location": ["Patna"],

    "size": [1800],

    "rooms": [3],

    "age": [5]

})


print("\nNew House:")

print(new_house)


# ---------------------------------------
# 11. Predict New House Price
# ---------------------------------------

predicted_price = pipeline.predict(
    new_house
)


print("\n==============================")
print("HOUSE PRICE PREDICTION")
print("==============================")

print(
    f"Predicted Price: ₹{predicted_price[0]:,.2f}"
)


# ---------------------------------------
# 12. Save Pipeline
# ---------------------------------------

joblib.dump(
    pipeline,
    "models/house_price_model.pkl"
)

print(
    "\nPipeline saved successfully!"
)