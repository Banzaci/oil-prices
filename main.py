"""
Oil Price Prediction Project

This script loads historical Brent oil prices, trains ML models,
and evaluates their performance using regression metrics.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create folders if they do not already exist.
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

df_oil = pd.read_csv((DATA_DIR / "BrentOilPrices.csv"))
df_usd = pd.read_csv((DATA_DIR / "USD_index.csv"))

df_usd["Change %"] = df_usd["Change %"].str.replace("%", "").astype(float)

df_usd["Date"] = df_usd["Date"].str.replace('"', '')
df_usd["Date"] = pd.to_datetime(df_usd["Date"])

df_oil["Date"] = pd.to_datetime(df_oil["Date"], format="mixed")
df_oil = df_oil.sort_values("Date")

df_oil["days"] = (df_oil["Date"] - df_oil["Date"].min()).dt.days

df = pd.merge(df_oil, df_usd, on="Date", how="inner")

df[["Price_x", "Price_y"]].corr()

X = df[["Price_y", "Change %", "days"]]
y = df["Price_x"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

fig, ax1 = plt.subplots()

# Oljepris (vänster axel)
# ax1.plot(df["Date"], df["Price_x"], color="blue", label="Oil Price")
# ax1.set_ylabel("Oil Price", color="blue")
# ax1.tick_params(axis="y", labelcolor="blue")

# # USD-index (höger axel)
# ax2 = ax1.twinx()
# ax2.plot(df["Date"], df["Price_y"], color="red", label="USD Index")
# ax2.set_ylabel("USD Index", color="red")
# ax2.tick_params(axis="y", labelcolor="red")

# plt.title("Oil Price vs USD Index")
# plt.show()

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

print("Predicted vs Actual Oil Prices:")
for pred, actual in zip(y_pred[:10], y_test[:10]):
    print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}") 

# print(f"Prediction for new data: {model.predict([[110, 0.5, 9000]])[0]:.2f}")
