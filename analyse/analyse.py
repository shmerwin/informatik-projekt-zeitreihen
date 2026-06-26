# Datenanalyse - Passantenfrequenzen Bad Nauheim

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# 1. Daten laden
# Pfad relativ zu dieser Datei
datei = Path(__file__).parent / "data" / "sensordata.json"
with open(datei, encoding="utf-8") as f:
    roh = json.load(f)

df = pd.DataFrame(roh["sensordata"])
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)


# 2. Daten anzeigen
print("Form (Zeilen, Spalten):", df.shape)
print("Spalten:", list(df.columns))

print("\nErste Zeilen:")
print(df.head())

print("\nStatistik der Zahlenspalten:")
print(df.describe())

print("\nFehlende Werte je Spalte:")
print(df.isna().sum())


# 3. Duplikate finden und bereinigen
anzahl = df.groupby(["name", "timestamp"]).size()
print("\nDuplikate (gleicher Sensor + gleiche Stunde):")
print("  betroffene Kombinationen:", int((anzahl > 1).sum()))
print("  maximal je Kombination:  ", int(anzahl.max()))

# konkretes Beispiel
name0, zeit0 = anzahl[anzahl > 1].index[0]
print("\nBeispiel-Duplikat (" + name0 + "):")
print(df[(df["name"] == name0) & (df["timestamp"] == zeit0)]
      [["name", "timestamp", "visitors", "avgDuration"]])

# (Sensor, Stunde) zu einer Zeile zusammenfassen
df = df.groupby(["name", "timestamp"], as_index=False).agg(
    visitors=("visitors", "median"),
    avgDuration=("avgDuration", "mean"),
)
print("\nNach Bereinigung:", df.shape)


# 4. Kennzahlen ausgeben
print("Sensoren: ", df["name"].nunique())
print("Zeitraum: ", df["timestamp"].min().date(), "bis", df["timestamp"].max().date())
print("Messpunkte gesamt: ", len(df))
print("Besucher min/median/mittel/max:",
      int(df["visitors"].min()), "/",
      int(df["visitors"].median()), "/",
      round(df["visitors"].mean(), 1), "/",
      int(df["visitors"].max()))

# Stärkste Standorte (Durchschnitt pro Stunde)
ranking = df.groupby("name")["visitors"].mean().sort_values(ascending=False)
print("\nRanking der Standorte (Durchschnitt Besucher/Stunde):")
print(ranking.head(30).round(1))

# Hoechster Einzelwert
spitze = df.loc[df["visitors"].idxmax()]
print("\nHoechster Einzelwert:", int(spitze["visitors"]), "Besucher")
print("  am", spitze["timestamp"], "bei", spitze["name"])


# 5. Visualisierung
# 5.1 Gesamtaufkommen über die Zeit
gesamt = df.groupby("timestamp")["visitors"].sum()
plt.figure(figsize=(12, 4))
plt.plot(gesamt.index, gesamt.values)
plt.title("Gesamtes Passantenaufkommen über die Zeit")
plt.xlabel("Zeit")
plt.ylabel("Besucher (Summe aller Sensoren)")
plt.tight_layout()
plt.show()

# 5.2 Durchschnitt pro Stunde
stunde = df.groupby(df["timestamp"].dt.hour)["visitors"].mean()
plt.figure(figsize=(8, 4))
plt.bar(stunde.index, stunde.values)
plt.title("Durchschnitt Besucher je Stunde")
plt.xlabel("Stunde")
plt.ylabel("Durchschnitt Besucher")
plt.tight_layout()
plt.show()

# 5.3 Wochenmuster
tage = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
woche = df.groupby(df["timestamp"].dt.dayofweek)["visitors"].mean()
plt.figure(figsize=(8, 4))
plt.bar([tage[i] for i in woche.index], woche.values)
plt.title("Wochenmuster - Durchschnitt Besucher je Wochentag")
plt.ylabel("Durchschnitt Besucher")
plt.tight_layout()
plt.show()

# 5.4 Standort-Ranking
plt.figure(figsize=(9, 8))
plt.barh(ranking.index[::-1], ranking.values[::-1])
plt.title("Standorte nach mittlerem Aufkommen")
plt.xlabel("Durchschnitt Besucher")
plt.tight_layout()
plt.show()

# 5.5 Beispiel-Zeitreihe des staerksten Standorts
top_name = ranking.index[0]
reihe = df[df["name"] == top_name].set_index("timestamp")["visitors"]
plt.figure(figsize=(12, 4))
plt.plot(reihe.index, reihe.values, color="red")
plt.title("Zeitreihe Beispiel-Standort: " + top_name)
plt.xlabel("Zeit")
plt.ylabel("Besucher")
plt.tight_layout()
plt.show()
