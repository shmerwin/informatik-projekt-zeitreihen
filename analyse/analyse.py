# Datenanalyse: Passantenfrequenzen Bad Nauheim

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# 1. Daten laden
# Pfad relativ zu dieser Datei
datei = Path(__file__).parent.parent / "data" / "raw" / "sensordata.json"
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
if (anzahl > 1).sum() > 0:
    name0, zeit0 = anzahl[anzahl > 1].index[0]
    print("\nBeispiel-Duplikat (" + name0 + "):")
    print(df[(df["name"] == name0) & (df["timestamp"] == zeit0)]
          [["name", "timestamp", "visitors", "avgDuration"]])
else:
    print("\nKein Beispiel-Duplikat möglich, da die Daten bereits 100% duplikatfrei sind!")

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

# Höchster Einzelwert
spitze = df.loc[df["visitors"].idxmax()]
print("\nHöchster Einzelwert:", int(spitze["visitors"]), "Besucher")
print("  am", spitze["timestamp"], "bei", spitze["name"])


# 5. Visualisierung
# Gesamtaufkommen über die Zeit
gesamt = df.groupby("timestamp")["visitors"].sum()
plt.figure(figsize=(12, 4))
plt.plot(gesamt.index, gesamt.values)
plt.title("Gesamtes Passantenaufkommen über die Zeit")
plt.xlabel("Zeit")
plt.ylabel("Besucher (Summe aller Sensoren)")
plt.tight_layout()
plt.show()

# Durchschnitt pro Stunde
stunde = df.groupby(df["timestamp"].dt.hour)["visitors"].mean()
plt.figure(figsize=(8, 4))
plt.bar(stunde.index, stunde.values)
plt.title("Durchschnitt Besucher je Stunde")
plt.xlabel("Stunde")
plt.ylabel("Durchschnitt Besucher")
plt.tight_layout()
plt.show()

# Wochenmuster
tage = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
woche = df.groupby(df["timestamp"].dt.dayofweek)["visitors"].mean()
plt.figure(figsize=(8, 4))
plt.bar([tage[i] for i in woche.index], woche.values)
plt.title("Wochenmuster - Durchschnitt Besucher je Wochentag")
plt.ylabel("Durchschnitt Besucher")
plt.tight_layout()
plt.show()

# Standort-Ranking
plt.figure(figsize=(9, 8))
plt.barh(ranking.index[::-1], ranking.values[::-1])
plt.title("Standorte nach mittlerem Aufkommen")
plt.xlabel("Durchschnitt Besucher")
plt.tight_layout()
plt.show()

# Beispiel-Zeitreihe des stärksten Standorts
top_name = ranking.index[0]
reihe = df[df["name"] == top_name].set_index("timestamp")["visitors"]
plt.figure(figsize=(12, 4))
plt.plot(reihe.index, reihe.values, color="red")
plt.title("Zeitreihe Beispiel-Standort: " + top_name)
plt.xlabel("Zeit")
plt.ylabel("Besucher")
plt.tight_layout()
plt.show()


# Schaut ob avgDuration bereits mit visitors korreliert oder unabhängige Informationen liefert
korr_visitors_dauer = df.groupby("name").apply(
    lambda g: g["visitors"].corr(g["avgDuration"])
).mean()
print(f"mittlere Korrelation visitors~avgDuration: {korr_visitors_dauer:.2f} "
      f"({'niedrig -> avgDuration trägt eigene Information' if korr_visitors_dauer < 0.8 else 'hoch -> unnötog zu visitors'})")

''' 
Korrelation zwischen den Standorten, einmal roh und einmal nach
Abzug des Tages-/Wochenendprofils je Sensor (sonst wird nur der gemeinsame
Tagesrhythmus gemessen, nicht echte gemeinsame Schwankungen/Anomalien)
'''
breit = df.pivot(index="timestamp", columns="name", values="visitors").interpolate().dropna()
profil = breit.groupby([breit.index.hour, breit.index.dayofweek >= 5]).transform("mean")
residuen = breit - profil


def poolen_kennzahlen(daten, bezeichnung):
    dreieck = daten.corr().where(np.triu(np.ones((daten.shape[1],) * 2), k=1).astype(bool)).stack()
    n_komponenten_80 = int(np.argmax(
        np.cumsum(PCA().fit(StandardScaler().fit_transform(daten)).explained_variance_ratio_) >= 0.8
    )) + 1
    print(f"{bezeichnung}: mittlere Korrelation {dreieck.mean():.2f}, "
          f"Anteil Paare |r|>0.5 {(dreieck.abs() > 0.5).mean():.0%}, "
          f"Komponenten für 80% Varianz {n_komponenten_80}/{daten.shape[1]}")
    return (dreieck.abs() > 0.5).mean(), n_komponenten_80


anteil_roh, komp_roh = poolen_kennzahlen(breit, "Rohdaten")
anteil_res, komp_res = poolen_kennzahlen(residuen, "rhythmusbereinigte Residuen")

print(f"\n(Rohdaten-Korrelation fällt von {anteil_roh:.0%} auf {anteil_res:.0%} starke Paare, "
      f"sobald der gemeinsame Tagesrhythmus rausgerechnet ist. Der Rest ist die tatsächliche "
      f"räumliche Kopplung zwischen Standorten.)")

print("\nFazit:")
print(f"avgDuration als 2. Feature:  {'ja' if korr_visitors_dauer < 0.8 else 'nein'}")
print(f"Standorte gemeinsam poolen:  "
      f"{'ja' if anteil_res > 0.3 and komp_res < breit.shape[1] * 0.3 else 'nein'}")
