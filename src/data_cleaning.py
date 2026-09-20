import pandas as pd

# -----------------------------------
# 1. Load Raw Dataset
# -----------------------------------

df = pd.read_csv("data/raw/house_data_raw.csv")

print("Raw Dataset Loaded Successfully!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# -----------------------------------
# 2. Check Missing Values
# -----------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------------
# 3. Check Duplicate Rows
# -----------------------------------

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# -----------------------------------
# 4. Rename Columns
# -----------------------------------

df.columns = [
    "location",
    "size",
    "rooms",
    "age",
    "house_price"
]


# -----------------------------------
# 5. Remove Duplicate Rows
# -----------------------------------

df = df.drop_duplicates()


# -----------------------------------
# 6. Check Invalid Values
# -----------------------------------

print("\nInvalid Size:")
print(df[df["size"] <= 0])

print("\nInvalid Rooms:")
print(df[df["rooms"] <= 0])

print("\nInvalid Age:")
print(df[df["age"] < 0])

print("\nInvalid House Price:")
print(df[df["house_price"] <= 0])


# -----------------------------------
# 7. Save Clean Dataset
# -----------------------------------

df.to_csv(
    "data/processed/house_data_clean.csv",
    index=False
)

print("\nClean dataset saved successfully!")

print("\nFinal Shape:")
print(df.shape)