# Anomalieerkennung Bad Nauheim

Das Projekt untersucht auffällige Passantenfrequenzen mit einem
LSTM-Forecaster, einem Transformer, einem LSTM-Autoencoder und einer saisonalen
24-Stunden-Baseline. Trainierte Modelle, Benchmarkergebnisse und der fertige
Projektbericht sind im Repository enthalten. Die großen Roh- und
Preprocessingdateien werden bei Bedarf über die städtische API neu erzeugt.

## Voraussetzungen und Installation

Das Projekt ist für Python 3.12 und die Versionen aus `requirements.txt`
vorgesehen.

Unter Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Unter Linux oder WSL:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Das Training funktioniert auf der CPU. Eine NVIDIA-GPU unter Linux oder WSL
kann die Trainingszeit verkürzen; ihre Einrichtung ist nicht projektspezifisch
und wird in der
[TensorFlow-Installationsanleitung](https://www.tensorflow.org/install/pip#linux)
beschrieben.

## Vollständiger Ablauf

### 1. Daten herunterladen und aufbereiten

```powershell
python -m src.download_data
python -m src.preprocessing
```

Dabei entstehen:

- `data/raw/sensordata.json`: unveränderte API-Daten,
- `data/processed/cleaned_sensordata.json`: stündlich zusammengefasste und
  aufbereitete Sensordaten.

Beide Dateien sind wegen ihrer Größe nicht im Git-Repository enthalten. Der
Standardzeitraum und die zeitlichen Teilmengen stehen zentral in
`src/data_config.py`. Abweichende Abrufgrenzen können mit `--since` und
`--until` an `src.download_data` übergeben werden.

### 2. Modelle trainieren

```powershell
python -m src.forecaster.train
python -m src.transformer.train
python -m src.autoencoder.train
python -m src.seasonal_baseline
```

Die Ausgaben liegen in den gleichnamigen Unterordnern von `saved_models/`:

- `forecaster.keras`, `transformer.keras` oder `autoencoder.keras`,
- `model_bundle.json` mit Skalierung, Modellkonfiguration und Datenaufteilung,
- `training_history.png` für die neuronalen Modelle.

Ein erneutes Training ersetzt das jeweilige Modell und entfernt die zuvor
kalibrierte Alarmregel aus dessen Bundle. Deshalb muss danach immer der
Benchmark erneut ausgeführt werden.

### 3. Alarmregeln kalibrieren und Modelle vergleichen

```powershell
python -m src.benchmark
```

Der Benchmark verwendet synthetische Anomalien, wählt für jedes Verfahren die
Alarmregel und schreibt sie in dessen `model_bundle.json`. Erst danach dürfen
Forecaster oder Transformer erneut ausgewertet werden.

Die wichtigsten Benchmarkausgaben sind:

- `saved_models/benchmark/model_comparison.csv`: kompakter Modellvergleich,
- `saved_models/benchmark/model_comparison.json`: vollständiger Vergleich,
- `saved_models/benchmark/roc_pr_comparison.png`: ROC- und PR-Kurven,
- `saved_models/<modell>/injection_benchmark.csv`: Ergebnisse je
  Anomalietyp, Stärke und Zufallsstartwert.

Die synthetischen Kennzahlen prüfen die bekannten Injektionen. Sie sind keine
Precision-, Recall- oder F1-Werte für unbekannte reale Ereignisse.

### 4. Kandidaten im Testzeitraum erzeugen

```powershell
python -m src.forecaster.evaluate
python -m src.forecaster.evaluate --model-dir saved_models/transformer
```

Für LSTM und Transformer entstehen jeweils im Modellordner:

| Datei | Inhalt |
|---|---|
| `anomalies.csv` | Stundenweise Scores, Kandidaten und fehlende Messungen |
| `anomaly_events.csv` | Zusammenhängende Forecast- und Datenqualitätsereignisse |
| `evaluation_metrics.json` | Prognosemetriken, Schwellen und Ereigniszahlen |
| `anomalies_example.png` | Beispielausschnitt mit Prognoseintervall und Kandidaten |
| `event_review_template.csv` | Vorlage zur manuellen Prüfung der Ereignisse |
| `labels_template.csv` | Vorlage für manuelle Punktlabels |

Die Einträge sind statistische Kandidaten und ohne fachliche Prüfung keine
bestätigten realen Anomalien. Autoencoder und saisonale Baseline werden im
gemeinsamen synthetischen Benchmark verglichen, erzeugen aber keinen eigenen
Kandidatenexport für den nicht injizierten Testzeitraum.

## Datenanalyse und Berichtsgrafiken

```powershell
python analyse/analyse.py
python analyse/preprocessing_plots.py
```

`analyse.py` beschreibt die Rohdaten. `preprocessing_plots.py` erzeugt die im
Bericht verwendeten Daten- und Preprocessingabbildungen unter
`Dokumentation/Figures/`. Beide Befehle benötigen die zuvor erzeugten
Datendateien.

Der Bericht liegt unter `Dokumentation/thesis.pdf`. Mit einer LaTeX-Installation
einschließlich Biber lässt er sich neu bauen:

```powershell
cd Dokumentation
latexmk -pdf -interaction=nonstopmode -halt-on-error thesis.tex
```


## Datenquelle und Reproduzierbarkeit

Die Daten stammen aus dem
[Smart-City-Cockpit Bad Nauheim](https://smartes.bad-nauheim.de/#!/map/pedestrianFrequencyCQM).
Nach Angaben des Portals werden von Smartphones ausgesendete Funkfrequenzen
anonym erfasst; `visitors` ist daher ein Proxy und keine direkte
Personenzählung. Ein späterer API-Abruf kann historische Änderungen enthalten
und muss nicht bitgleich mit dem für die gespeicherten Modelle verwendeten
Abrufstand sein. Vor einer Weiterverwendung der städtischen Daten ist außerdem
deren Lizenzstatus zu klären.
