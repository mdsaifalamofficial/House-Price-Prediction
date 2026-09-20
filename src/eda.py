import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------
# 1. Load Clean Dataset
# -----------------------------------

df = pd.read_csv(
    "data/processed/house_data_clean.csv"
)

print("Dataset Loaded Successfully!")


# -----------------------------------
# 2. Basic Information
# -----------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)


# -----------------------------------
# 3. Statistical Summary
# -----------------------------------

print("\nStatistical Summary:")
print(df.describe())


# -----------------------------------
# 4. Average Values
# -----------------------------------

print("\nAverage Size:")
print(df["size"].mean())

print("\nAverage Rooms:")
print(df["rooms"].mean())

print("\nAverage Age:")
print(df["age"].mean())

print("\nAverage Price:")
print(df["house_price"].mean())


# -----------------------------------
# 5. Minimum and Maximum
# -----------------------------------

print("\nMinimum Size:")
print(df["size"].min())

print("\nMaximum Size:")
print(df["size"].max())

print("\nMinimum Price:")
print(df["house_price"].min())

print("\nMaximum Price:")
print(df["house_price"].max())


# -----------------------------------
# 6. Location Count
# -----------------------------------

print("\nHouse Count by Location:")
print(df["location"].value_counts())


# -----------------------------------
# 7. Average Price by Location
# -----------------------------------

print("\nAverage Price by Location:")

location_price = (
    df.groupby("location")["house_price"]
    .mean()
)

print(location_price)


# -----------------------------------
# 8. Size vs Price
# -----------------------------------

plt.scatter(
    df["size"],
    df["house_price"]
)

plt.xlabel("House Size (sqft)")
plt.ylabel("House Price")
plt.title("House Size vs House Price")

plt.show()


# -----------------------------------
# 9. Rooms vs Price
# -----------------------------------

plt.scatter(
    df["rooms"],
    df["house_price"]
)

plt.xlabel("Number of Rooms")
plt.ylabel("House Price")
plt.title("Rooms vs House Price")

plt.show()


# -----------------------------------
# 10. Age vs Price
# -----------------------------------

plt.scatter(
    df["age"],
    df["house_price"]
)

plt.xlabel("House Age")
plt.ylabel("House Price")
plt.title("House Age vs House Price")

plt.show()


# -----------------------------------
# 11. Location vs Price
# -----------------------------------

location_price.plot(
    kind="bar"
)

plt.xlabel("Location")
plt.ylabel("Average House Price")
plt.title("Average House Price by Location")

plt.xticks(rotation=45)

plt.show()


# -----------------------------------
# 12. Correlation
# -----------------------------------

correlation = df[
    ["size", "rooms", "age", "house_price"]
].corr()

print("\nCorrelation Matrix:")
print(correlation)


# -----------------------------------
# 13. Correlation Heatmap
# -----------------------------------

sns.heatmap(
    correlation,
    annot=True
)

plt.title("Correlation Heatmap")

plt.show()