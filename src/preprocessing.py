import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


TRAIN_END = pd.Timestamp("2025-07-08", tz="UTC")
VAL_END = pd.Timestamp("2025-07-10", tz="UTC")


def process_and_impute_full_data(filepath):
    # JSON-Datei einlesen
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_json = json.load(f)

    df = pd.DataFrame(raw_json['sensordata'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Statische Infos (lon, lat, locationId etc.) pro Sensor sichern
    meta_cols = [c for c in df.columns if c not in ['timestamp', 'visitors', 'avgDuration']]
    sensor_meta = df[meta_cols].drop_duplicates(subset=['name'])

    # Pivot-Tabelle: (Zeit Index, Sensoren Spalten)
    # Sensor lücken separat füllen!
    df_pivot = df.pivot_table(index='timestamp', columns='name',
                              values=['visitors', 'avgDuration'], aggfunc='mean')

    # Auf vollen Stundentakt auffüllen (erzeugt NaNs bei Lücken)
    df_pivot = df_pivot.resample('1h').asfreq()

    # Nur die tatsächliche Laufzeit jedes Sensors gilt als aktiv.
    sensor_ranges = df.groupby('name')['timestamp'].agg(['min', 'max'])
    active = pd.DataFrame(False, index=df_pivot.index, columns=df_pivot.columns)
    for feature, name in df_pivot.columns:
        active[(feature, name)] = (
            (df_pivot.index >= sensor_ranges.loc[name, 'min'])
            & (df_pivot.index <= sensor_ranges.loc[name, 'max'])
        )
    measured = df_pivot.where(active)
    was_missing = measured.isna() & active

    # 1. Nur vollständig kurze Lücken interpolieren, getrennt je Datensplit.
    def is_short_gap(series):
        missing = series.isna()
        gap = missing.ne(missing.shift()).cumsum()
        return missing & missing.groupby(gap).transform('sum').le(2)

    short_gaps = measured.apply(is_short_gap)
    df_pivot = measured.copy()
    split_masks = [
        df_pivot.index < TRAIN_END,
        (df_pivot.index >= TRAIN_END) & (df_pivot.index < VAL_END),
        df_pivot.index >= VAL_END,
    ]
    for split in split_masks:
        part = df_pivot.loc[split]
        linear = part.interpolate(method='linear', limit_area='inside')
        df_pivot.loc[split] = part.where(~short_gaps.loc[split], linear)

    # 2. Größere Lücken nur mit gemessenen Trainingswerten füllen.
    local_time = df_pivot.index.tz_convert('Europe/Berlin')
    train = df_pivot.index < TRAIN_END
    reference = measured.loc[train]
    train_time = local_time[train]

    weekday_hour_mean = reference.groupby([train_time.weekday, train_time.hour]).mean()
    keys = pd.MultiIndex.from_arrays([local_time.weekday, local_time.hour])
    weekday_hour_values = weekday_hour_mean.reindex(keys)
    weekday_hour_values.index = df_pivot.index
    df_pivot = df_pivot.fillna(weekday_hour_values)

    hour_mean = reference.groupby(train_time.hour).mean()
    hour_values = hour_mean.reindex(local_time.hour)
    hour_values.index = df_pivot.index
    df_pivot = df_pivot.fillna(hour_values).fillna(reference.mean()).where(active)

    # Tabelle wieder "flach" machen
    df_clean = df_pivot.stack(level='name')
    df_clean['is_imputed'] = was_missing.stack(level='name').any(axis=1)
    df_clean = df_clean.reset_index().dropna(
        subset=['visitors', 'avgDuration'], how='all')
    df_clean['visitors'] = df_clean['visitors'].round()
    df_clean['avgDuration'] = df_clean['avgDuration'].round(2)

    # Statische Infos (Koordinaten etc.) wieder anfügen
    df_final = df_clean.merge(sensor_meta, on='name', how='left')
    return df_final


def create_windows(data, window_size):
    # Zeitreihe im Windows zerschneiden
    X = []
    for i in range(len(data) - window_size):
        window = data[i:(i + window_size)]
        X.append(window)
    return np.array(X)


def preprocess_data(filepath, window_size=24):
    # 1. Lücken füllen
    df_full = process_and_impute_full_data(filepath)

    # 2. Clean Data als JSON speichern
    processed_dir = Path(filepath).parent.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "cleaned_sensordata.json"

    # Zeitstempel für den Export
    df_export = df_full.copy()
    df_export['timestamp'] = df_export['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')

    export_dict = {"sensordata": df_export.to_dict(orient='records')}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(export_dict, f, indent=2)

    # 3. Passanten summieren
    df_model = df_full.groupby('timestamp')['visitors'].sum().reset_index()

    # 4. Daten skalieren (0 bis 1)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_model[['visitors']])

    # 5. Fenster erstellen (Samples, Timesteps, Features)
    X_train = create_windows(df_scaled, window_size)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    return X_train, scaler, df_model


if __name__ == "__main__":
    # Kurzer Testlauf
    datei_pfad = Path(__file__).parent.parent / "data" / "raw" / "sensordata.json"

    if datei_pfad.exists():
        X_test, train_scaler, df_clean = preprocess_data(datei_pfad, window_size=24)
        print(f"Vorverarbeitung erfolgreich. Shape der Matrix: {X_test.shape}")
        print("Vollständige JSON gespeichert unter: data/processed/cleaned_sensordata.json")
    else:
        print(f"Fehler: Datei nicht gefunden unter {datei_pfad}")
