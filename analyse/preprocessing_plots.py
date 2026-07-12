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


# Rohdaten laden
with open(BASIS / "data" / "raw" / "sensordata.json", encoding="utf-8") as f:
    roh = json.load(f)
df = pd.DataFrame(roh["sensordata"])
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)


# Bereinigungsschritte nachvollziehen (wie src/preprocessing.py),
# um für jede Stunde die Herkunft ihres Wertes zu kennen
pivot = df.pivot_table(index="timestamp", columns="name",
                       values=["visitors", "avgDuration"], aggfunc="mean")
pivot = pivot.resample("1h").asfreq()  # volles Stundenraster, Lücken = NaN

fehlt = pivot["visitors"].isna()

schritt1 = pivot.interpolate(method="linear", limit=2)
interpoliert = fehlt & schritt1["visitors"].notna()

wochentag = schritt1.index.weekday
stunde = schritt1.index.hour
schritt2 = schritt1.groupby([wochentag, stunde]).transform(lambda x: x.fillna(x.mean()))
muster = schritt1["visitors"].isna() & schritt2["visitors"].notna()

schritt3 = schritt2.groupby(stunde).transform(lambda x: x.fillna(x.mean()))
notfall = schritt2["visitors"].isna() & schritt3["visitors"].notna()

zeitraster = pivot.index
sensoren = fehlt.sum().sort_values(ascending=False).index  # nach Lückenanteil sortiert

print("Stunden im Raster:", len(zeitraster), "| Sensoren:", len(sensoren))
print("Fehlende Stunden :", int(fehlt.sum().sum()),
      "davon interpoliert:", int(interpoliert.sum().sum()),
      "| Muster:", int(muster.sum().sum()),
      "| Notfall:", int(notfall.sum().sum()))


# Abbildung 1: Lücken vor und nach der Bereinigung (Heatmap)
fig, achsen = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

vorher = np.array([fehlt[s].values for s in sensoren])
achsen[0].imshow(vorher, aspect="auto", interpolation="nearest",
                 cmap=ListedColormap(["whitesmoke", "firebrick"]))
achsen[0].set_title("Vor der Bereinigung: fehlende Stunden je Sensor (rot)")

# 0 = echt, 1 = interpoliert, 2 = Muster, 3 = Notfall
nachher = np.zeros(vorher.shape)
for j, s in enumerate(sensoren):
    nachher[j, interpoliert[s].values] = 1
    nachher[j, muster[s].values] = 2
    nachher[j, notfall[s].values] = 3
achsen[1].imshow(nachher, aspect="auto", interpolation="nearest", vmin=0, vmax=3,
                 cmap=ListedColormap(["white", "blue", "orange", "red"]))
achsen[1].set_title("Nach der Bereinigung: Herkunft der ergänzten Werte")

for ax in achsen:
    ax.set_yticks(range(len(sensoren)))
    ax.set_yticklabels(sensoren, fontsize=6.5)
xticks = np.arange(0, len(zeitraster), 24)
achsen[1].set_xticks(xticks)
achsen[1].set_xticklabels([zeitraster[i].strftime("%d.%m.") for i in xticks],
                          rotation=45, fontsize=8)
achsen[1].set_xlabel("Datum")

legende = [plt.Rectangle((0, 0), 1, 1, color=f, ec="gray")
           for f in ["white", "blue", "orange", "red"]]
achsen[1].legend(legende,
                 ["echter Messwert",
                  "linear interpoliert (Lücke bis 2 h)",
                  "Mittel aus gleichem Wochentag und gleicher Stunde",
                  "Mittel aus gleicher Stunde (Notfallstufe)"],
                 loc="upper center", bbox_to_anchor=(0.5, -0.28),
                 ncol=2, fontsize=9, frameon=False)
speichern(fig, "preproc_luecken_heatmap.png")


# Abbildung 2: zwei Beispiel-Zeitreihen (typischer Sensor und Extremfall)
beispiele = [sensoren[len(sensoren) // 2], sensoren[0]]
fig, achsen = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
for ax, s in zip(achsen, beispiele):
    werte = schritt3["visitors"][s]
    mi = interpoliert[s].values
    mm = (muster[s] | notfall[s]).values
    ax.plot(zeitraster, werte, lw=0.9, color="gray", label="Zeitreihe nach Bereinigung")
    ax.scatter(zeitraster[mi], werte[mi], s=18, color="steelblue",
               zorder=3, label="linear interpoliert")
    ax.scatter(zeitraster[mm], werte[mm], s=18, color="orange",
               zorder=3, label="aus Tagesmuster ergänzt")
    ax.set_title(f"{s} ({fehlt[s].mean() * 100:.0f} % der Stunden ergänzt)")
    ax.set_ylabel("Passanten pro Stunde")
    ax.legend(loc="upper right", fontsize=8)
achsen[1].set_xlabel("Datum")
speichern(fig, "preproc_beispielreihen.png")


# Abbildung 3: Wie lang sind die Lücken? (begründet die 2h-Grenze)
laengen = pd.concat(
    [(~fehlt[s]).cumsum()[fehlt[s]].value_counts() for s in sensoren])

# in Gruppen einteilen, damit jeder Balken gleich breit ist
# (die Längen reichen von 1 h bis 298 h, fast alle sind aber kurz)
gruppen = pd.cut(laengen, bins=[0, 1, 2, 6, 12, 24, 48, 320],
                 labels=["1 h", "2 h", "3-6 h", "7-12 h",
                         "13-24 h", "25-48 h", "> 48 h"])
anzahl = gruppen.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(anzahl.index.astype(str), anzahl.values, color="blue", edgecolor="black")
ax.axvline(1.5, color="red", ls="--", lw=1.5)  # Grenze zwischen 2 h und 3-6 h
ax.text(1.65, anzahl.max() * 0.85,
        "bis 2 h: lineare Interpolation\nab 3 h: Füllung aus Tagesmustern",
        fontsize=9, color="red")
ax.set_xlabel("Lückenlänge")
ax.set_ylabel("Anzahl Lücken")
ax.set_title(f"Länge der {len(laengen)} Messlücken im Rohdatensatz")
speichern(fig, "preproc_lueckenlaengen.png")


# Abbildung 4: Anteil ergänzter Stunden je Sensor (gestapelte Balken)
anteile = pd.DataFrame({
    "linear interpoliert": interpoliert.sum(),
    "Wochentag-Stunden-Mittel": muster.sum(),
    "Stunden-Mittel (Notfallstufe)": notfall.sum(),
}).loc[sensoren[::-1]] / len(zeitraster) * 100 # größter Anteil oben

fig, ax = plt.subplots(figsize=(9, 8))
anteile.plot.barh(stacked=True, ax=ax, width=0.8,
                  color=["blue", "orange", "red"])
ax.set_xlabel("Anteil ergänzter Stunden in %")
ax.set_ylabel("")
ax.set_title("Wie viel jeder Sensorreihe ergänzt werden musste")
ax.tick_params(axis="y", labelsize=7)
ax.legend(loc="lower right", fontsize=9)
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
