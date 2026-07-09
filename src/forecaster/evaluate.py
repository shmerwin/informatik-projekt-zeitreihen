'''
Auswertung des gespeicherten Forecasters auf dem Testzeitraum (ab 10.07.)
Schwellenwert: 99%-Quantil der Vorhersagefehler auf der Validierung,
je Kanal getrennt (avgDuration ist deutlich verrauschter als visitors
ein gemeinsamer Schwellenwert würde vom Rauschen dominiert).
Eine Stunde gilt als Anomalie, wenn mindestens ein Kanal darüber liegt.
'''
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import models

from train import FEATURES, MODEL_DIR, load_data, make_windows

THRESHOLD_QUANTILE = 0.99

model = models.load_model(os.path.join(MODEL_DIR, "forecaster.keras"))

df = load_data()
data = make_windows(df)

# Vorhersagefehler je Kanal
val_errors = np.abs(model.predict(data["val"]["X"], verbose=0)
                    - data["val"]["y"])
test_errors = np.abs(model.predict(data["test"]["X"], verbose=0)
                     - data["test"]["y"])

thresholds = np.quantile(val_errors, THRESHOLD_QUANTILE, axis=0)
print("Schwellenwerte je Kanal (skaliert):",
      dict(zip(FEATURES, thresholds.round(3))))

results = pd.DataFrame({
    "sensor": data["test"]["sensor"],
    "time": data["test"]["time"],
})
for i, feature in enumerate(FEATURES):
    results[f"error_{feature}"] = test_errors[:, i]
results["anomaly"] = (test_errors > thresholds).any(axis=1)

print(f"\nAnomalien im Testzeitraum: {int(results['anomaly'].sum())} "
      f"von {len(results)} Stunden ({results['anomaly'].mean():.1%})")

print("\nSensoren mit den meisten Anomalien:")
print(results[results["anomaly"]].groupby("sensor").size()
      .sort_values(ascending=False).head(5).to_string())

results.to_csv(os.path.join(MODEL_DIR, "anomalies.csv"), index=False)
print("\nGespeichert:", os.path.join(MODEL_DIR, "anomalies.csv"))

# Beispielplot: Sensor mit den meisten Anomalien
top_sensor = results[results["anomaly"]].groupby("sensor").size().idxmax()
series = (df[df["name"] == top_sensor]
          .drop_duplicates("timestamp")
          .set_index("timestamp")["visitors"])
hits = results[(results["sensor"] == top_sensor) & results["anomaly"]]
test_start = results["time"].min()

plt.figure(figsize=(12, 4))
plt.plot(series[series.index >= test_start], label="visitors")
plt.scatter(hits["time"], series.reindex(hits["time"]).values,
            color="red", zorder=3, label="Anomalie")
plt.title(f"Forecaster-Anomalien im Testzeitraum: {top_sensor}")
plt.xlabel("Zeit")
plt.ylabel("Besucher")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "anomalies_example.png"), dpi=150)
plt.show()
