import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def process_and_impute_full_data(filepath):
    # JSON-Datei einlesen
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_json = json.load(f)

    df = pd.DataFrame(raw_json['sensordata'])
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

    # Statische Infos (lon, lat, locationId etc.) pro Sensor sichern
    meta_cols = [c for c in df.columns if c not in ['timestamp', 'visitors', 'avgDuration']]
    sensor_meta = df[meta_cols].drop_duplicates(subset=['name'])

    # Pivot-Tabelle: (Zeit Index, Sensoren Spalten)
    # Sensor lücken separat füllen!
    df_pivot = df.pivot_table(index='timestamp', columns='name',
                              values=['visitors', 'avgDuration'], aggfunc='mean')

    # Auf vollen Stundentakt auffüllen (erzeugt NaNs bei Lücken)
    df_pivot = df_pivot.resample('1h').asfreq()

    # 1. Kleine Lücken interpolieren (max 2h)
    df_pivot = df_pivot.interpolate(method='linear', limit=2)

    # 2. Größere Lücken mit Wochentag+Stunde füllen
    weekday = df_pivot.index.weekday
    hour = df_pivot.index.hour
    df_pivot = df_pivot.groupby([weekday, hour]).transform(lambda x: x.fillna(x.mean()))

    # 3. Notfall-Lösung: Stunden-Durchschnitt
    if df_pivot.isna().sum().sum() > 0:
        df_pivot = df_pivot.groupby(hour).transform(lambda x: x.fillna(x.mean()))

    # Tabelle wieder "flach" machen
    df_clean = df_pivot.stack(level='name').reset_index()
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