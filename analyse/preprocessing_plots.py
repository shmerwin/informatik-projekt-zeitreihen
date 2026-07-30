# Plots zum Preprocessing
# Erzeugt die Grafiken für den Bericht und legt sie direkt in
# Dokumentation/Figures/ ab, damit LaTeX sie einbinden kann.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BASIS = Path(__file__).parent.parent
FIGURES = BASIS / "Dokumentation" / "Figures"
FIGURES.mkdir(exist_ok=True)


def speichern(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("gespeichert:", name)


# Rohdaten und Ergebnis der zentralen Preprocessing-Pipeline laden
with open(BASIS / "data" / "raw" / "sensordata.json", encoding="utf-8") as f:
    roh = json.load(f)
df = pd.DataFrame(roh["sensordata"])
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

with open(BASIS / "data" / "processed" / "cleaned_sensordata.json",
          encoding="utf-8") as f:
    clean = pd.DataFrame(json.load(f)["sensordata"])
clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True)

zeitraster = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="1h")
sensor_names = sorted(clean["name"].unique())
aktiv = pd.DataFrame(False, index=zeitraster, columns=sensor_names)
fehlt = aktiv.copy()
for name, sensor in clean.groupby("name"):
    aktiv.loc[sensor["timestamp"], name] = True
    fehlt.loc[sensor["timestamp"], name] = sensor["is_imputed"].to_numpy(bool)

anteil = fehlt.sum() / aktiv.sum()
sensoren = anteil.sort_values(ascending=False).index

print("Aktive Sensorstunden:", int(aktiv.sum().sum()),
      "| davon ergänzt:", int(fehlt.sum().sum()))


# Abbildung 1: Lücken vor und nach der Bereinigung (Heatmap)
fig, achsen = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

vorher = np.full((len(sensoren), len(zeitraster)), 2)
for j, s in enumerate(sensoren):
    vorher[j, aktiv[s].values] = 0
    vorher[j, fehlt[s].values] = 1
achsen[0].imshow(vorher, aspect="auto", interpolation="nearest",
                 vmin=0, vmax=2,
                 cmap=ListedColormap(["white", "firebrick", "lightgray"]))
achsen[0].set_title("Vor der Bereinigung: Messlücken innerhalb der aktiven Zeit")

nachher = np.full(vorher.shape, 2)
for j, s in enumerate(sensoren):
    nachher[j, aktiv[s].values] = 0
    nachher[j, fehlt[s].values] = 1
achsen[1].imshow(nachher, aspect="auto", interpolation="nearest", vmin=0, vmax=2,
                 cmap=ListedColormap(["white", "steelblue", "lightgray"]))
achsen[1].set_title("Nach der Bereinigung: gemessene und ergänzte Stunden")

for ax in achsen:
    ax.set_yticks(range(len(sensoren)))
    ax.set_yticklabels(sensoren, fontsize=6.5)
xticks = np.arange(0, len(zeitraster), 24)
achsen[1].set_xticks(xticks)
lokales_raster = zeitraster.tz_convert("Europe/Berlin")
achsen[1].set_xticklabels([lokales_raster[i].strftime("%d.%m.") for i in xticks],
                          rotation=45, fontsize=8)
achsen[1].set_xlabel("Lokales Datum (Europe/Berlin)")

legende = [plt.Rectangle((0, 0), 1, 1, color=f, ec="gray")
           for f in ["white", "steelblue", "lightgray"]]
achsen[1].legend(legende,
                 ["gemessener Wert", "ergänzter Wert", "Sensor inaktiv"],
                 loc="upper center", bbox_to_anchor=(0.5, -0.28),
                 ncol=3, fontsize=9, frameon=False)
speichern(fig, "preproc_luecken_heatmap.png")


# Abbildung 2: typische und am stärksten ergänzte Sensorreihe
beispiele = [sensoren[len(sensoren) // 2], anteil.idxmax()]
fig, achsen = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
for ax, s in zip(achsen, beispiele):
    sensor = clean[clean["name"] == s].set_index("timestamp")
    zeit = sensor.index.tz_convert("Europe/Berlin")
    ax.plot(zeit, sensor["visitors"], lw=0.9, color="gray",
            label="bereinigte Zeitreihe")
    mi = sensor["is_imputed"].to_numpy(bool)
    ax.scatter(zeit[mi], sensor.loc[mi, "visitors"], s=18, color="steelblue",
               zorder=3, label="ergänzte Stunde")
    ax.set_title(f"{s}: {len(sensor)} aktive Stunden, {mi.mean() * 100:.1f} % ergänzt")
    ax.set_ylabel("Passanten pro Stunde")
    ax.legend(loc="upper right", fontsize=8)
achsen[1].set_xlabel("Lokale Zeit (Europe/Berlin)")
speichern(fig, "preproc_beispielreihen.png")


# Abbildung 3: Wie lang sind die Lücken
laengen = pd.concat(
    [(~fehlt[s]).cumsum()[fehlt[s]].value_counts() for s in sensoren])

# in Gruppen einteilen, damit jeder Balken gleich breit ist
# Lange Endausfälle bleiben sichtbar statt aus der Grafik herauszufallen.
gruppen = pd.cut(laengen, bins=[0, 1, 2, 6, 12, 24, np.inf],
                 labels=["1 h", "2 h", "3-6 h", "7-12 h",
                         "13-24 h", ">24 h"])
anzahl = gruppen.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(anzahl.index.astype(str), anzahl.values, color="blue", edgecolor="black")
ax.axvline(1.5, color="red", ls="--", lw=1.5)  # Grenze zwischen 2 h und 3-6 h
ax.text(1.65, anzahl.max() * 0.85,
        "erste 2 h jeder Lücke: Fortschreibung\nRest: Trainingsprofile",
        fontsize=9, color="red")
ax.set_xlabel("Lückenlänge")
ax.set_ylabel("Anzahl Lücken")
ax.set_title(f"Länge der {len(laengen)} Messlücken im Rohdatensatz")
speichern(fig, "preproc_lueckenlaengen.png")


# Abbildung 4: Anteil ergänzter Stunden je aktiver Sensorperiode
anteile = (anteil.loc[sensoren[::-1]] * 100)
fig, ax = plt.subplots(figsize=(9, 8))
anteile.plot.barh(ax=ax, width=0.8, color="steelblue")
ax.set_xlabel("Anteil ergänzter aktiver Stunden in %")
ax.set_ylabel("")
ax.set_title("Imputationsanteil innerhalb der aktiven Sensorzeit")
ax.tick_params(axis="y", labelsize=7)
speichern(fig, "preproc_imputationsanteil.png")


# Abbildung 5: Redundanzen im Rohdatensatz
messungen = df.groupby(["name", "timestamp"]).size()
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(messungen.values, bins=range(1, 12), edgecolor="black",
        align="left", color="blue")
ax.set_xlabel("Messungen für denselben Sensor in derselben Stunde")
ax.set_ylabel("Häufigkeit")
ax.set_title("Redundanzen im Rohdatensatz: oft mehrere Messungen pro Stunde")
ax.set_xticks(range(1, 11))
speichern(fig, "preproc_mehrfachmessungen.png")

print("\nAlle Abbildungen liegen in:", FIGURES)
