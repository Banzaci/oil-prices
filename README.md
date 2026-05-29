# Förutsägelse av Brent-oljepriset med linjär regression

**Kurs:** Pythonprogrammering för AI-utveckling
**Kurskod:** YKPYT26V2

---

# 1. Problemformulering

Målet med projektet är att förutsäga Brent-oljepriset (USD per barrel) utifrån:

* USD-indexets värde
* Daglig procentuell förändring i USD-indexet

Eftersom oljepriset är ett kontinuerligt numeriskt värde är detta ett **regressionsproblem**.

---

# 2. Dataset

Två dataset användes:

## BrentOilPrices.csv

Dagliga stängningspriser för Brent-råolja mellan **1987–2022**.

## USD_index.csv

Dagliga värden för det amerikanska dollarindexet (DXY), inklusive:

* USD-indexets värde
* Daglig procentuell förändring (`Change %`)

---

# 3. Varför dessa dataset?

Dollar och olja har historiskt ett tydligt omvänt samband eftersom olja handlas globalt i USD.

En starkare dollar gör oljan dyrare för länder med andra valutor, vilket ofta leder till:

* lägre efterfrågan
* pressade oljepriser

Dataseten är:

* publika
* välstrukturerade
* omfattar över 35 års historik

Det ger ett bra underlag för att undersöka om sambandet kan modelleras linjärt.

---

# 4. Dataförberedelse

Följande steg genomfördes:

## Rensning av data

* `%`-tecknet togs bort från kolumnen `Change %`
* värden konverterades till `float`
* citationstecken i datumkolumnen togs bort
* datum konverterades till `datetime`

## Sortering och feature engineering

Oljedatasetet sorterades kronologiskt.

En ny feature skapades:

```python
days
```

Den representerar antal dagar sedan:

```text
1987-05-20
```

Syftet är att ge modellen ett numeriskt tidsmått för att fånga långsiktiga trender.

## Sammanfogning av dataset

Dataseten slogs samman med:

```python
inner join
```

på kolumnen:

```python
Date
```

Det innebär att endast datum som finns i båda dataset inkluderas.

## Saknade värden

Inga saknade värden hittades efter join-operationen.

---

# 5. Features och target

## Features

* `usd_price`
* `Change %`
* `days`

## Target

* `oil_price`

---

# 6. Modellval

## Vald modell

```python
Projektet använder både linjär regression och Random Forest Regression.
```

## Motivering

Linjär regression valdes eftersom:

* problemet är regression, inte klassificering
* modellen är lätt att tolka
* koefficienterna visar direkt hur varje feature påverkar oljepriset
* modellen fungerar som en bra baseline innan mer avancerade modeller testas

---

# 7. Träning och testdata

Datasetet delades upp enligt:

* 80 % träning
* 20 % test

```python
random_state=42
```

användes för reproducerbarhet.

Ingen normalisering användes eftersom vanlig linjär regression utan regularisering inte påverkas av skillnader i skala mellan features.

---

# 8. Resultat

## Modellens prestanda

| Mått | Värde             | Tolkning                            |
| ---- | ----------------- | ----------------------------------- |
| MSE  | 373.96            | Genomsnittligt kvadratiskt fel      |
| RMSE | ≈ 19.3 USD/barrel | Modellen missar i snitt med ~19 USD |
| R²   | 0.569             | Förklarar ca 57 % av variationen    |

---

## Tolkning av resultaten

Ett R2-värde på:

```text
0.57
```

innebär att modellen förklarar drygt hälften av variationen i oljepriset.

RMSE på cirka:

```text
19 USD/barrel
```

är relativt högt eftersom oljepriset under perioden varierade mellan ungefär:

```text
9–133 USD/barrel
```

---

## Jämförelse med Random Forest Regression

En Random Forest-modell testades också för att jämföra resultaten med linjär regression.

Random Forest gav ett RMSE-värde på ungefär:

```text
1.74 USD/barrel
```

vilket var betydligt lägre än den linjära regressionens:

```text
19.3 USD/barrel
```

Det tyder på att sambandet mellan variablerna inte är helt linjärt och att mer flexibla modeller kan fånga mönster som linjär regression missar.


# 9. Modellkoefficienter

| Feature   | Koefficient | Tolkning                         |
| --------- | ----------- | -------------------------------- |
| Change %  | +2.37       | Starkaste featuren               |
| usd_price | −1.85       | Starkare dollar → lägre oljepris |
| days      | +0.004      | Svag långsiktig positiv trend    |

## Kommentar

Det negativa sambandet mellan USD och oljepris följer ekonomisk teori.

Den positiva koefficienten för `Change %` kan bero på att kortsiktiga USD-rörelser samvarierar med volatilitet snarare än långsiktiga trender.

---

# 10. Scenariotester

Tre manuella scenarier testades med:

```python
days = 9000
```

(cirka år 2012)

| Scenario                 | Prediktion     |
| ------------------------ | -------------- |
| Stark USD (115, +1.0 %)  | ~24 USD/barrel |
| Svag USD (95, −1.0 %)    | ~56 USD/barrel |
| Neutral USD (105, 0.0 %) | ~40 USD/barrel |

## Analys

Resultaten följer rätt riktning:

```text
svagare dollar → högre oljepris
```

men modellen underskattar kraftigt de verkliga prisnivåerna runt 2012 då Brentpriset ofta låg mellan:

```text
100–120 USD/barrel
```

---

# 11. Begränsningar i modellen

## Exempel från verkligheten

Den:

```text
2022-11-14
```

var det faktiska Brentpriset:

```text
93.14 USD/barrel
```

Modellen förutsåg endast:

```text
57.01 USD/barrel
```

Fel:

```text
≈ 36 USD
```

Under 2022 påverkades oljepriset kraftigt av:

* Rysslands invasion av Ukraina
* geopolitisk osäkerhet
* energikris

Dessa faktorer finns inte med i modellen.

---

# 12. Residualanalys

Residualplotten visar tydliga mönster:

## Höga oljepriser (80–100 USD)

* stora positiva fel
* modellen underskattar ofta

## Mellannivåer (40–70 USD)

* relativt jämna fel
* modellen fungerar bättre

## Låga prisnivåer (0–20 USD)

* systematiskt positiva residualer
* modellen underskattar även här

---

Residualanalysen visar att felets storlek varierar beroende på prisnivå, vilket indikerar att modellen inte fångar sambandet fullt ut.

---

# 13. Förbättringsförslag

## Fler features

Exempel:

* OPEC-produktionsbeslut
* råoljelager (EIA-data)
* inflationsjusterade priser
* geopolitiska händelser

---

## Feature engineering

Exempel:

* laggade värden
* glidande medelvärden
* volatilitet

---

## Alternativa modeller

Möjliga modeller:

* Random Forest Regression
* Gradient Boosting
* XGBoost

Dessa kan fånga icke-linjära samband bättre.

---

## Tidsseriemodeller

Mer lämpliga modeller för tidsdata:

* ARIMA
* Prophet
* LSTM

---

## Korrekt train/test-split

Nuvarande random split kan leda till dataläckage.

En bättre metod är:

```text
träna: 1987–2015
testa: 2016–2022
```

---

# 14. Reflektion

Modellen fungerar bra som ett pedagogiskt exempel på:

* regression
* feature engineering
* dataförberedelse
* modellutvärdering

Däremot är modellen inte tillräckligt tillförlitlig för verkliga handelsbeslut.

Finansiella tidsserier påverkas av:

* geopolitik
* ekonomiska kriser
* pandemier
* marknadspsykologi

vilket gör problemet betydligt mer komplext än vad en enkel linjär modell kan fånga.

---

# 15. Slutsats

Projektet visar att det finns ett mätbart samband mellan:

* USD-indexet
* oljepriset

men sambandet är långt ifrån tillräckligt starkt för att skapa träffsäkra prognoser med enbart linjär regression.

Mer avancerade modeller och fler features krävs för realistiska prediktioner.
