import pandas as pd

# Load raw dataset
df = pd.read_csv("data/raw/house_data_raw.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display number of rows and columns
print("\nShape:")
print(df.shape)

# Display column names
print("\nColumns:")
print(df.columns)

# Display data types
print("\nData Types:")
print(df.dtypes)