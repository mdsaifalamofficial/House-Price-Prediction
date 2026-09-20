import pandas as pd


# 1. Load clean dataset
df = pd.read_csv(
    "data/processed/house_data_clean.csv"
)

print("Clean dataset loaded successfully!")


# 2. Basic information
print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)


# 3. Separate features and target
X = df[
    [
        "location",
        "size",
        "rooms",
        "age"
    ]
]

y = df["house_price"]


print("\nFeatures X:")
print(X.head())

print("\nTarget y:")
print(y.head())


# 4. One-Hot Encoding
X_encoded = pd.get_dummies(
    X,
    columns=["location"],
    dtype=int
)


# 5. Display encoded data
print("\nEncoded Features:")
print(X_encoded.head())


# 6. Compare shapes
print("\nOriginal X Shape:")
print(X.shape)

print("\nEncoded X Shape:")
print(X_encoded.shape)


# 7. Display encoded columns
print("\nEncoded Columns:")
print(X_encoded.columns)


# 8. Create ML-ready dataset
ml_ready_df = X_encoded.copy()

ml_ready_df["house_price"] = y


# 9. Display final dataset
print("\nML Ready Dataset:")
print(ml_ready_df.head())


# 10. Save feature-engineered dataset
ml_ready_df.to_csv(
    "data/processed/house_data_features.csv",
    index=False
)


print("\nFeature-engineered dataset saved successfully!")