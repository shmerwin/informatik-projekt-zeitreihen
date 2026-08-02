"""Erzeugt die beiden Anomaliegrafiken der Präsentation aus anomalies.csv."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "saved_models" / "forecaster"


def load_candidates():
    data = pd.read_csv(MODEL_DIR / "anomalies.csv")
    data["time"] = pd.to_datetime(data["time"], utc=True)
    for column in ["forecast_anomaly", "missing_measurement"]:
        data[column] = data[column].astype(str).str.lower().eq("true")
    return data.sort_values(["sensor", "time"])


def _window(data, sensor, center, before, after):
    return data[
        data["sensor"].eq(sensor)
        & data["time"].between(
            center - pd.Timedelta(hours=before),
            center + pd.Timedelta(hours=after),
        )
    ]


def _panel(axis, data, points, feature, title, subtitle):
    axis.fill_between(
        data["time"], data[f"lower_{feature}"], data[f"upper_{feature}"],
        color="#9ecae1", alpha=0.55, label="kalibriertes Intervall",
    )
    axis.plot(data["time"], data[f"actual_{feature}"], "o-", ms=3,
              color="#0b4f7c", label="Ist")
    axis.plot(data["time"], data[f"forecast_{feature}"],
              color="#1b9e77", lw=2, label="Medianprognose")
    axis.scatter(points["time"], points[f"actual_{feature}"], s=45,
                 color="#e6007e", zorder=3, label="markierter Punkt")
    axis.set_ylabel(feature)
    axis.set_title(f"{title}\n{subtitle}", loc="left", fontsize=10)
    axis.grid(alpha=0.2)


def save_diagnosis(data):
    direct = data[
        data["sensor"].eq("Kurpark - Eingang Sprudelhof")
        & data["forecast_anomaly"] & data["trigger_features"].eq("visitors")
    ].nlargest(1, "score_visitors")
    duration = data[data["actual_avgDuration"].notna()].nlargest(
        1, "score_avgDuration"
    )
    event_id = data.loc[
        data["sensor"].eq("Rosengarten") & data["forecast_anomaly"],
        "forecast_event_id",
    ].mode().iat[0]
    event = data[data["forecast_event_id"].eq(event_id)]

    row = direct.iloc[0]
    boundary = row["lower_visitors"] if (
        row["actual_visitors"] < row["lower_visitors"]
    ) else row["upper_visitors"]
    relation = "unter" if row["actual_visitors"] < boundary else "über"
    examples = [
        (_window(data, row["sensor"], row["time"], 10, 10), direct, "visitors",
         f"{row['sensor']} · visitors",
         f"Direkter Alarm: Ist {row['actual_visitors']:.0f}, "
         f"{relation} Intervallgrenze {boundary:.2f}"),
        (_window(data, duration.iloc[0]["sensor"], duration.iloc[0]["time"], 6, 2),
         duration, "avgDuration", f"{duration.iloc[0]['sensor']} · avgDuration",
         f"Stärkste Abweichung: Ist {duration.iloc[0]['actual_avgDuration']:.2f}; "
         "dient nur als Kontext"),
        (_window(data, event.iloc[0]["sensor"], event["time"].min(), 8,
                 int((event["time"].max() - event["time"].min()).total_seconds() / 3600) + 6),
         event, "visitors", f"{event.iloc[0]['sensor']} · visitors",
         f"Zusammenhängendes Ereignis mit {len(event)} Alarmstunden"),
    ]

    figure, axes = plt.subplots(3, 1, figsize=(14, 8))
    for axis, example in zip(axes, examples):
        _panel(axis, *example)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
                  ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = MODEL_DIR / "diagnosis_examples.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def save_alert_overview(data):
    sensors = pd.Index(data["sensor"].drop_duplicates())
    times = pd.date_range(data["time"].min(), data["time"].max(), freq="1h")
    index = pd.MultiIndex.from_product([sensors, times], names=["sensor", "time"])
    grid = data.assign(present=True).drop_duplicates(["sensor", "time"]).set_index(
        ["sensor", "time"]
    ).reindex(index)
    state = np.full(len(grid), 3, dtype=int)
    state[grid["present"].fillna(False)] = 0
    state[grid["missing_measurement"].fillna(False)] = 1
    state[grid["forecast_anomaly"].fillna(False)] = 2

    figure, axis = plt.subplots(figsize=(14, 7))
    axis.imshow(
        state.reshape(len(sensors), len(times)), aspect="auto",
        interpolation="nearest", vmin=-0.5, vmax=3.5,
        cmap=ListedColormap(["white", "#6baed6", "#e6007e", "#d9d9d9"]),
    )
    axis.set_yticks(np.arange(len(sensors)), sensors, fontsize=7)
    ticks = np.arange(0, len(times), 24)
    axis.set_xticks(ticks, [times[i].strftime("%d.%m.") for i in ticks])
    axis.set(xlabel="Testzeitraum", ylabel="Sensor")
    axis.legend(handles=[
        Patch(facecolor="white", edgecolor="#bbbbbb", label="gemessen / unauffällig"),
        Patch(facecolor="#6baed6", label="Messung fehlt"),
        Patch(facecolor="#e6007e", label="Forecast-Kandidat"),
        Patch(facecolor="#d9d9d9", label="außerhalb verfügbarer Fenster"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False)
    figure.tight_layout()
    path = MODEL_DIR / "alert_overview.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main():
    data = load_candidates()
    outputs = [save_diagnosis(data), save_alert_overview(data)]
    assert all(path.exists() and path.stat().st_size for path in outputs)
    print("Gespeichert:", *(str(path) for path in outputs), sep="\n")


if __name__ == "__main__":
    main()
