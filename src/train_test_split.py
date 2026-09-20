import pandas as pd

from sklearn.model_selection import train_test_split


# 1. Load feature-engineered dataset
df = pd.read_csv(
    "data/processed/house_data_features.csv"
)

print("Feature dataset loaded successfully!")


# 2. Display dataset
print("\nDataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# 3. Separate features and target
X = df.drop(
    "house_price",
    axis=1
)

y = df["house_price"]


# 4. Display X and y
print("\nFeatures X:")
print(X.head())

print("\nX Shape:")
print(X.shape)

print("\nTarget y:")
print(y.head())

print("\ny Shape:")
print(y.shape)


# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. Display split shapes
print("\nX_train Shape:")
print(X_train.shape)

print("\nX_test Shape:")
print(X_test.shape)

print("\ny_train Shape:")
print(y_train.shape)

print("\ny_test Shape:")
print(y_test.shape)


print("\nTrain/Test Split completed successfully!")