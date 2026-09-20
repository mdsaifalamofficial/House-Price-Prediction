import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Load feature-engineered dataset
df = pd.read_csv(
    "data/processed/house_data_features.csv"
)

print("Dataset loaded successfully!")


# 2. Separate features and target
X = df.drop(
    "house_price",
    axis=1
)

y = df["house_price"]


# 3. Train/Test Split
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


# 4. Create Linear Regression model
model = LinearRegression()


# 5. Train the model
model.fit(
    X_train,
    y_train
)

print("\nModel training completed!")


# 6. Make predictions
predictions = model.predict(
    X_test
)


# 7. Display predictions
print("\nPredicted Prices:")
print(predictions)


# 8. Display actual prices
print("\nActual Prices:")
print(y_test.values)


# 9. Actual vs Predicted comparison
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\nActual vs Predicted:")
print(comparison.round(2))


# 10. Model coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Coefficients:")
print(coefficients)


# 11. Intercept
print("\nModel Intercept:")
print(model.intercept_)


# 12. Mean Absolute Error
mae = mean_absolute_error(
    y_test,
    predictions
)

print("\nMean Absolute Error:")
print(mae)


# 13. R² Score
r2 = r2_score(
    y_test,
    predictions
)

print("\nR² Score:")
print(r2)