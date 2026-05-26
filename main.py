from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# LOAD DATA
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Läser in två CSV-filer: historiska Brent-oljepriser och USD-index (dollarns styrka).
df_oil = pd.read_csv(DATA_DIR / "BrentOilPrices.csv")
df_usd = pd.read_csv(DATA_DIR / "USD_index.csv")

# -------------------------
# CLEAN USD DATA
# -------------------------
# "Change %" innehåller procenttecken som sträng, t.ex. "1.5%".
# Vi tar bort "%" och omvandlar till float så att kolumnen kan användas som feature.
df_usd["Change %"] = df_usd["Change %"].str.replace("%", "").astype(float)

# Datumkolumnen kan innehålla citationstecken, t.ex. '"2020-01-01"'.
# Vi tar bort dem och omvandlar sedan till datetime-format för att kunna merga på datum.
df_usd["Date"] = df_usd["Date"].str.replace('"', '')
df_usd["Date"] = pd.to_datetime(df_usd["Date"])

# -------------------------
# CLEAN OIL DATA
# -------------------------
# format="mixed" låter pandas hantera blandade datumformat i samma kolumn.
df_oil["Date"] = pd.to_datetime(df_oil["Date"], format="mixed")

# Sorterar raderna i stigande datumordning (äldst → nyast).
df_oil = df_oil.sort_values("Date")

# Skapar en numerisk feature "days" = antal dagar sedan det tidigaste datumet i datasetet.
# Det ger modellen ett mått på tid, eftersom datum i sig inte är numeriska.
df_oil["days"] = (df_oil["Date"] - df_oil["Date"].min()).dt.days

# -------------------------
# MERGE
# -------------------------
# Slår ihop de två dataseten på "Date"-kolumnen.
# how="inner" innebär att bara rader med datum som finns i BÅDA filerna tas med.
df = pd.merge(df_oil, df_usd, on="Date", how="inner")

# Efter merge heter priskolumnerna "Price_x" (olja) och "Price_y" (USD-index).
# Vi döper om dem till tydligare namn för att undvika förvirring.
df = df.rename(columns={
    "Price_x": "oil_price",
    "Price_y": "usd_price"
})

# -------------------------
# FEATURES / TARGET
# -------------------------
# Features (X) är de variabler modellen ska lära sig från:
#   - usd_price:  USD-indexets värde (dollarns styrka)
#   - Change %:   procentuell daglig förändring i USD-indexet
#   - days:       antal dagar sedan startdatum (fångar tidstrender)
# Target (y) är det vi vill förutsäga: oljepriset.
X = df[["usd_price", "Change %", "days"]]
y = df["oil_price"]

# -------------------------
# TRAIN / TEST SPLIT
# -------------------------
# Delar upp datat i 80% träning och 20% test.
# random_state=42 ger ett reproducerbart resultat varje körning.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# MODEL
# -------------------------
# Linjär regression försöker hitta den räta linje (hyperplan i 3D) som
# minimerar summan av kvadratiska fel mot träningsdatat. (Regression eftersom jag förutsäger ett kontinuerligt pris.)
model = LinearRegression()
model.fit(X_train, y_train)  # Modellen tränas på träningsdata

# -------------------------
# PREDICTION
# -------------------------
# Kör modellen på testdatan för att få förutsägelser.
# y_pred innehåller de förutspådda oljepriserna.
y_pred = model.predict(X_test)

# Hjälpfunktion för att testa enskilda scenarion.
# Skapar en DataFrame med samma kolumnnamn som träningsdata, vilket krävs av sklearn.
def predict_oil_price(usd_price, change_pct, days):
    X_new = pd.DataFrame([[usd_price, change_pct, days]],
                         columns=["usd_price", "Change %", "days"])
    return model.predict(X_new)[0]

# Tre scenarion: stark dollar, svag dollar, neutral.
# En stark dollar brukar historiskt sett korrelera med lägre oljepriser (och vice versa).
print("Strong USD:", predict_oil_price(115, 1.0, 9000))
print("Weak USD:", predict_oil_price(95, -1.0, 9000))
print("Neutral:", predict_oil_price(105, 0.0, 9000))

# -------------------------
# EVALUATION
# -------------------------
mse = mean_squared_error(y_test, y_pred) # Mäter hur stora felen är i genomsnitt, i kvadrat. Lämpar sig för att straffa stora fel mer än små.
rmse = np.sqrt(mse)  # RMSE är i samma enhet som oljepriset (USD/barrel)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
# r2 visar hur stor andel av variationen i oljepriset som modellen förklarar.
# 1.0 = perfekt, 0.0 = modellen är lika bra som att alltid gissa medelvärdet.
print("R2 Score:", r2)

print("\nPredicted vs Actual (first 10):")
results = pd.DataFrame({
    "Predicted": y_pred,
    "Actual": y_test.values # y_test innehåller de riktiga (verkliga) värdena från datasetet
})
print(results.head(10)) # 10 staplar

# -------------------------
# VISUALISATION
# -------------------------
# Scatter: förutsagt värde på x-axeln, felet (residualen) på y-axeln.
# Om modellen vore perfekt skulle alla punkter ligga längs den röda linjen (fel = 0).
# Slumpmässigt spridda punkter tyder på att felen är oberoende av varandra — det är bra.
# Mönster (t.ex. en bågform) tyder på att linjär regression missar något systematiskt.
residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals, s=5)
plt.axhline(0, color="red")
plt.title("Residuals (Prediction Errors)")
plt.xlabel("Predicted Oil Price")
plt.ylabel("Error (Actual - Predicted)")
plt.show()

# Scatter: faktiskt vs förutsagt oljepris.
# Punkter nära den diagonala linjen innebär att modellen träffar bra.
plt.figure()
plt.scatter(y_test, y_pred, s=5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=1)
plt.xlabel("Actual Oil Price")
plt.ylabel("Predicted Oil Price")
plt.title("Actual vs Predicted Oil Price")
plt.show()

# Koefficienterna visar hur mycket oljepriset förändras (i USD) när en feature
# ökar med 1 enhet, givet att övriga features hålls konstanta.
# Absolut värde avgör "styrka" — positivt/negativt tecken avgör riktning.
print("\nFeature importance (sorted by absolute coefficient):")
importance = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_
})
importance = importance.sort_values("coefficient", key=abs, ascending=False)
print(importance)