import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

df = pd.read_csv(
    "data/processed/house_data_features.csv"
)

print("Dataset loaded successfully!")


# -----------------------------------
# 2. Separate Features and Target
# -----------------------------------

X = df.drop(
    "house_price",
    axis=1
)

y = df["house_price"]


print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# -----------------------------------
# 3. Train/Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# -----------------------------------
# 4. Linear Regression
# -----------------------------------

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(
    X_test
)


# -----------------------------------
# 5. Random Forest
# -----------------------------------

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_predictions = random_forest_model.predict(
    X_test
)


# -----------------------------------
# 6. Evaluation Function
# -----------------------------------

def evaluate_model(
    model_name,
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# -----------------------------------
# 7. Evaluate Both Models
# -----------------------------------

linear_results = evaluate_model(
    "Linear Regression",
    y_test,
    linear_predictions
)

random_forest_results = evaluate_model(
    "Random Forest",
    y_test,
    random_forest_predictions
)


# -----------------------------------
# 8. Comparison Table
# -----------------------------------

results = pd.DataFrame([
    linear_results,
    random_forest_results
])


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    results.round(2)
)


# -----------------------------------
# 9. Actual vs Predicted
# -----------------------------------

comparison = pd.DataFrame({

    "Actual": y_test.values,

    "Linear Regression":
        linear_predictions,

    "Random Forest":
        random_forest_predictions

})


print("\n==============================")
print("ACTUAL VS PREDICTED")
print("==============================")

print(
    comparison.round(2)
)


# -----------------------------------
# 10. Linear Regression Errors
# -----------------------------------

linear_errors = (
    y_test.values
    - linear_predictions
)

print("\nLinear Regression Errors:")

print(
    linear_errors
)


# -----------------------------------
# 11. Random Forest Errors
# -----------------------------------

random_forest_errors = (
    y_test.values
    - random_forest_predictions
)

print("\nRandom Forest Errors:")

print(
    random_forest_errors
)


# -----------------------------------
# 12. Save Comparison
# -----------------------------------

results.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)

print(
    "\nModel comparison saved successfully!"
)