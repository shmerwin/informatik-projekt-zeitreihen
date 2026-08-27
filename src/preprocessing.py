"""Kausale Aufbereitung der stündlichen Sensordaten."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_config import TRAIN_END_UTC

TRAIN_END = pd.Timestamp(TRAIN_END_UTC)
FEATURES = ["visitors", "avgDuration"]


def _profil_auf_zeitachse(referenz, ziel_index, gruppe):
    """Überträgt ein ausschließlich aus Trainingsdaten gelerntes Profil."""
    lokale_referenz = referenz.index.tz_convert("Europe/Berlin")
    lokale_ziele = ziel_index.tz_convert("Europe/Berlin")

    if gruppe == "wochentag_stunde":
        profil = referenz.groupby(
            [lokale_referenz.weekday, lokale_referenz.hour]
        ).mean()
        schluessel = pd.MultiIndex.from_arrays(
            [lokale_ziele.weekday, lokale_ziele.hour]
        )
    else:
        profil = referenz.groupby(lokale_referenz.hour).mean()
        schluessel = lokale_ziele.hour

    ergebnis = profil.reindex(schluessel)
    ergebnis.index = ziel_index
    return ergebnis


def process_and_impute_full_data(filepath, train_end=TRAIN_END):
    """Bereinigt die Rohdaten, ohne zukünftige Messwerte zu verwenden."""
    with open(filepath, "r", encoding="utf-8") as datei:
        roh = json.load(datei)

    df = pd.DataFrame(roh["sensordata"])
    fehlend = {"timestamp", "name", *FEATURES} - set(df.columns)
    if fehlend:
        raise ValueError(f"Fehlende Pflichtfelder: {sorted(fehlend)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="raise")
        if not np.isfinite(df[feature]).all() or (df[feature] < 0).any():
            raise ValueError(
                f"{feature} enthält fehlende, unendliche oder negative Rohwerte"
            )

    # Metadaten sowie Anzahl der Mehrfachmessungen bleiben nachvollziehbar.
    meta_spalten = [
        spalte for spalte in df.columns
        if spalte not in ["timestamp", *FEATURES]
    ]
    sensor_meta = df[meta_spalten].drop_duplicates("name", keep="first")
    quellanzahl = (
        df.groupby(["timestamp", "name"]).size()
        .rename("source_record_count").reset_index()
    )

    # Mehrfachmessungen werden wie bisher gemittelt und auf ein Stundenraster gelegt.
    gemessen = df.pivot_table(
        index="timestamp", columns="name", values=FEATURES, aggfunc="mean"
    ).resample("1h").asfreq()

    # Wenn mindestens zehn Sensoren gleichzeitig exakt 0 Besucher und
    # 0 Verweildauer melden, handelt es sich um einen gemeinsamen Systemausfall.
    vorhanden = gemessen["visitors"].notna() & gemessen["avgDuration"].notna()
    nullmeldung = gemessen["visitors"].eq(0) & gemessen["avgDuration"].eq(0)
    systemausfall = vorhanden.sum(axis=1).ge(10) & (nullmeldung | ~vorhanden).all(axis=1)
    gemessen.loc[systemausfall] = np.nan

    starts = df.groupby("name")["timestamp"].min()
    aktiv = pd.DataFrame(False, index=gemessen.index, columns=gemessen.columns)
    for feature, sensor in gemessen.columns:
        # Ein Sensor aus dem aktuellen Karten-Layer bleibt nach seiner ersten
        # Messung aktiv; ein dauerhaft ausbleibender Stream ist ein Ausfall.
        aktiv[(feature, sensor)] = gemessen.index >= starts.loc[sensor]
    gemessen = gemessen.where(aktiv)

    werte = gemessen.copy()
    methode = pd.DataFrame("", index=werte.index, columns=werte.columns)
    methode[gemessen.notna()] = "observed"

    # Die ersten beiden Stunden jeder Lücke werden nur aus der Vergangenheit
    # fortgeschrieben. Anders als lineare Interpolation kann dies kein Ziel y[t]
    # in den Kontext X[t] zurückspiegeln.
    vorwaerts = gemessen.ffill(limit=2).where(aktiv)
    maske = aktiv & werte.isna() & vorwaerts.notna()
    werte[maske] = vorwaerts[maske]
    methode[maske] = "causal_ffill"

    # Alle weiteren Ersatzwerte stammen nur aus ursprünglich gemessenen
    # Trainingswerten, niemals aus Validation oder Test.
    referenz = gemessen.loc[gemessen.index < pd.Timestamp(train_end)]
    kandidaten = [
        ("weekday_hour_mean", _profil_auf_zeitachse(
            referenz, werte.index, "wochentag_stunde"
        )),
        ("hour_mean", _profil_auf_zeitachse(referenz, werte.index, "stunde")),
        ("sensor_mean", pd.DataFrame(
            np.tile(referenz.mean().to_numpy(), (len(werte), 1)),
            index=werte.index, columns=werte.columns,
        )),
    ]
    for name, kandidat in kandidaten:
        maske = aktiv & werte.isna() & kandidat.notna()
        werte[maske] = kandidat[maske]
        methode[maske] = name

    if (aktiv & werte.isna()).any().any():
        raise ValueError("Nicht alle aktiven Sensorstunden konnten imputiert werden")

    # Zurück in eine flache Tabelle. Beide Messgrößen fehlen in diesen Daten
    # immer gemeinsam; deshalb reicht eine gemeinsame Imputationsmaske.
    gestapelt = werte.stack(level="name", future_stack=True)
    gueltig = gestapelt[FEATURES].notna().any(axis=1)
    gestapelt = gestapelt.loc[gueltig]
    sauber = gestapelt.reset_index()
    beobachtet = gemessen.notna().stack(level="name", future_stack=True).loc[gueltig]
    methoden = methode.stack(level="name", future_stack=True).loc[gueltig]
    sauber["is_imputed"] = (~beobachtet.all(axis=1)).to_numpy()
    sauber["imputation_method"] = methoden[FEATURES[0]].to_numpy()
    sauber = sauber.merge(quellanzahl, on=["timestamp", "name"], how="left")
    sauber["source_record_count"] = sauber["source_record_count"].fillna(0).astype(int)
    sauber = sauber.merge(sensor_meta, on="name", how="left")
    sauber["visitors"] = sauber["visitors"].round()
    sauber["avgDuration"] = sauber["avgDuration"].round(2)
    return sauber


def save_processed_data(filepath, output_path=None):
    """Bereitet die Daten auf und schreibt standardkonformes JSON ohne NaN."""
    filepath = Path(filepath)
    output_path = Path(output_path or (
        filepath.parent.parent / "processed" / "cleaned_sensordata.json"
    ))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sauber = process_and_impute_full_data(filepath)

    export = sauber.copy()
    export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    # pandas ersetzt hier NaN aus optionalen Metadaten durch null.
    records = json.loads(export.to_json(orient="records"))
    with open(output_path, "w", encoding="utf-8") as datei:
        json.dump(
            {"sensordata": records}, datei, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        )
    return sauber


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    standard = Path(__file__).resolve().parent.parent / "data" / "raw" / "sensordata.json"
    parser.add_argument("--input", type=Path, default=standard)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sauber = save_processed_data(args.input, args.output)
    print(f"Vorverarbeitung erfolgreich: {len(sauber)} Sensorstunden, "
          f"davon {int(sauber['is_imputed'].sum())} imputiert.")


if __name__ == "__main__":
    main()
