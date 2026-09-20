import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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


# 4. Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# 5. Train the model
model.fit(
    X_train,
    y_train
)

print("\nRandom Forest training completed!")


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


# 9. Actual vs Predicted
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\nActual vs Predicted:")
print(comparison.round(2))


# 10. MAE
mae = mean_absolute_error(
    y_test,
    predictions
)

print("\nMean Absolute Error:")
print(mae)


# 11. R² Score
r2 = r2_score(
    y_test,
    predictions
)

print("\nR² Score:")
print(r2)


# 12. Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)