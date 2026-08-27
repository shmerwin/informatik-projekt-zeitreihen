# Anomalieerkennung Bad Nauheim

Die Pipeline lädt historische Passantenmessungen, bereitet sie kausal auf und
vergleicht LSTM-Forecaster, Transformer, LSTM-Autoencoder und eine saisonale
24-Stunden-Baseline mit demselben chronologischen Protokoll. Die beiden
Prognosemodelle kombinieren einen 24-Stunden-Kontext mit einer getrennten
Wochenreferenz der Stunde `t-168`.

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
python -m src.download_data
python -m src.preprocessing
```

Der für die aktuellen Modellläufe verwendete Zeitraum lässt sich mit festen
Grenzen erneut abrufen. Nachträgliche Änderungen historischer API-Daten können
dabei nicht ausgeschlossen werden; die gespeicherten Modellartefakte beziehen
sich auf den beim Trainingslauf vorliegenden Abrufstand:

```powershell
python -m src.download_data `
  --since 2025-06-30T00:00:00Z `
  --until 2026-07-01T00:00:00Z
python -m src.preprocessing
```

Nach Angaben des [Smart-City-Cockpits](https://smartes.bad-nauheim.de/#!/map/pedestrianFrequencyCQM)
werden von Smartphones ausgesendete Funkfrequenzen anonym erfasst und gezählt;
MAC-Adressen und personenbezogene Daten werden nicht ausgelesen. Die
[Datenstrategie der Stadt Bad Nauheim](https://www.bad-nauheim.de/frontend/0_dokumente/3_lebenswert/digitalisierung/Bad_Nauheim_Datenstrategie_1.0.pdf)
weist zugleich auf eine gewisse Unsicherheit bei wiederkehrenden Besuchern hin.
`visitors` ist daher ein aus Gerätesignalen abgeleiteter Proxy und keine direkte
Personen-Ground-Truth. Bei der Prüfung von Portal und API am 27.08.2026 wurde
keine ausdrückliche Nachnutzungs- oder Open-Data-Lizenz identifiziert;
technische Zugänglichkeit allein begründet kein Nachnutzungsrecht. Der
Lizenzstatus muss vor einer anderweitigen Weiterverwendung geklärt werden.

`data/raw/sensordata.json` und
`data/processed/cleaned_sensordata.json` sind generierte Dateien und wegen
ihrer Größe nicht Teil des Git-Repositories.
Der verifizierte Snapshot enthält 537.810 Rohzeilen. Nach der Aufbereitung
umfasst er 263.505 aktive Sensorstunden, davon 177.004 ursprünglich gemessen
und 86.501 imputiert.

Der unter `saved_models/legacy/2025-07/preprocessing` archivierte
Imputationsbenchmark dokumentiert eine explorative Vorprüfung vom 30.06. bis
07.07.2025. Er ist kein erneuter Gütenachweis für den später erweiterten
Jahresdatensatz und gehört nicht zu den aktuellen Ergebnisartefakten.

## Chronologische Aufteilung

- Training: vor dem 1. April 2026
- Tuning und Modellauswahl: April 2026
- Kalibrierung von Intervallen und Alarmregeln: Mai 2026
- chronologischer Test: Juni 2026

Die Grenzen liegen zentral in `src/data_config.py` und werden von
Preprocessing, Training, Benchmark und Evaluation gemeinsam verwendet.
Der Juni wurde retrospektiv als Testmonat gewählt, nachdem in einem zuvor
untersuchten Juli-Abruf neun Sensoren ohne Messung aufgefallen waren. Deshalb
ist er trotz der programmgesteuerten Trennung kein unberührt festgelegter
konfirmatorischer Holdout. Im Juni sind alle 30 Sensoren beobachtet; von 21.600
aktiven Sensorstunden sind 13.142 gemessen.

## Modelle neu erzeugen

```powershell
python -m src.forecaster.train
python -m src.transformer.train
python -m src.autoencoder.train
python -m src.seasonal_baseline
python -m src.benchmark
python -m src.forecaster.evaluate
python -m src.forecaster.evaluate --model-dir saved_models/transformer
```

Die neuronalen Modelle wählen Architektur und Epoche ausschließlich über den
Tuning-Zeitraum. Die kuratierten Kandidaten reichen von kleinen bis deutlich
größeren Netzen; Early Stopping begrenzt unnötige Epochen. Der
Autoencoder-Suchlauf prüft Architekturen bis höchstens 200 Epochen. Der
synthetische Benchmark wählt danach die Alarmregel mit synthetischen Anomalien im
Kalibrierungsblock und bewertet sie mit getrennten Zufallsstartwerten im
Testblock. Diese Kennzahlen sind keine Bewertung realer Anomalieereignisse.

LSTM-Forecaster und Transformer erhalten je Zielstunde ein `24 x 8`-Fenster,
eine Sensor-ID und zwölf Hilfsmerkmale: fünf Kalenderwerte, die beiden
skalierten Werte von `t-168` samt Beobachtungsflag sowie Mittelwert und
Standardabweichung beider skalierten Messgrößen im Fenster. Nur beim Training
wird, sofern ein ausreichend langer ursprünglich gemessener Block vorhanden
ist, ein zusammenhängender Block von einer bis vier Kontextstunden maskiert.
Bei 1,18 % der Prognosefenster fehlt ein geeigneter Block; sie bleiben
unverändert. Tuning, Kalibrierung, Test und Inferenz bleiben unverändert. Die
saisonale Baseline und die MASE-Skalierung verwenden weiterhin `t-24`.

## Tests

```powershell
python -m unittest discover -s tests -v
```
