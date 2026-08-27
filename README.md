# Anomalieerkennung Bad Nauheim

Die Pipeline lädt historische Passantenmessungen, bereitet sie kausal auf und
vergleicht LSTM-Forecaster, Transformer, LSTM-Autoencoder und eine saisonale
24-Stunden-Baseline mit demselben chronologischen Protokoll.

## Python-Umgebung

Unter Windows wird die projektlokale Python-3.12-Umgebung einmalig so erzeugt:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

TensorFlow ab Version 2.11 unterstützt NVIDIA-GPUs nicht mehr nativ unter
Windows. Für die RTX 3080 wird deshalb Ubuntu in WSL2 verwendet. Die
Linux-Umgebung liegt absichtlich im schnellen WSL-Dateisystem:

```bash
python3.12 -m venv ~/.venvs/anomalie
source ~/.venvs/anomalie/bin/activate
python -m pip install -r requirements.txt 'tensorflow[and-cuda]==2.21.0'
TF_PACKAGE_DIR=$(dirname "$(python -c 'import tensorflow as tf; print(tf.__file__)')")
ln -sf "$VIRTUAL_ENV"/lib/python3.12/site-packages/nvidia/*/lib/*.so* "$TF_PACKAGE_DIR"/
ln -sf "$VIRTUAL_ENV"/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas "$VIRTUAL_ENV/bin/ptxas"
```

Der Symlink-Schritt folgt der
[TensorFlow-Installationsanleitung](https://www.tensorflow.org/install/pip#linux).
Trainingsläufe werden in jeder neuen WSL-Sitzung mit folgenden Variablen
gestartet:

```bash
source ~/.venvs/anomalie/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/tensorflow"
export TF_FORCE_GPU_ALLOW_GROWTH=true
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -m src.transformer.train
```

Die Geräteprüfung muss mindestens ein `PhysicalDevice` vom Typ `GPU` ausgeben.

## Daten aktualisieren

Der Abruf verwendet nur die Python-Standardbibliothek, liest alle eindeutigen
Sensor-IDs aus der Karte und teilt lange Zeiträume in 60-Tage-Anfragen. Die
Zieldatei wird erst nach einem vollständigen erfolgreichen Abruf ersetzt. Ohne
Argumente verwendet der Befehl den festen Abrufzeitraum vom 30.06.2025 bis zum
01.07.2026 (Endzeitpunkt exklusiv).

```powershell
py -3.12 -m src.download_data
py -3.12 -m src.preprocessing
```

Der für die aktuellen Modellläufe verwendete Zeitraum lässt sich mit festen
Grenzen erneut abrufen. Nachträgliche Änderungen historischer API-Daten können
dabei nicht ausgeschlossen werden; die gespeicherten Modellartefakte beziehen
sich auf den beim Trainingslauf vorliegenden Abrufstand:

```powershell
py -3.12 -m src.download_data `
  --since 2025-06-30T00:00:00Z `
  --until 2026-07-01T00:00:00Z
py -3.12 -m src.preprocessing
```

`data/raw/sensordata.json` und
`data/processed/cleaned_sensordata.json` sind generierte Dateien und wegen
ihrer Größe nicht Teil des Git-Repositories.

Der Imputationsbenchmark unter `saved_models/preprocessing` dokumentiert eine
explorative Vorprüfung vom 30.06. bis 07.07.2025. Er ist kein erneuter
Gütenachweis für den später erweiterten Jahresdatensatz.

## Chronologische Aufteilung

- Training: vor dem 1. April 2026
- Tuning und Modellauswahl: April 2026
- Kalibrierung von Intervallen und Alarmregeln: Mai 2026
- chronologischer Test: Juni 2026

Die Grenzen liegen zentral in `src/data_config.py` und werden von
Preprocessing, Training, Benchmark und Evaluation gemeinsam verwendet.

## Modelle neu erzeugen

```powershell
py -3.12 -m src.forecaster.train
py -3.12 -m src.transformer.train
py -3.12 -m src.autoencoder.train
py -3.12 -m src.seasonal_baseline
py -3.12 -m src.benchmark
py -3.12 -m src.forecaster.evaluate
py -3.12 -m src.forecaster.evaluate --model-dir saved_models/transformer
```

Die neuronalen Modelle wählen Architektur und Epoche ausschließlich über den
Tuning-Zeitraum. Die kuratierten Kandidaten reichen von kleinen bis deutlich
größeren Netzen; Early Stopping begrenzt unnötige Epochen. Der synthetische
Benchmark wählt danach die Alarmregel mit synthetischen Anomalien im
Kalibrierungsblock und bewertet sie mit getrennten Zufallsstartwerten im
Testblock. Diese Kennzahlen sind keine Bewertung realer Anomalieereignisse.

## Tests

```powershell
py -3.12 -m unittest discover -s tests -v
```
