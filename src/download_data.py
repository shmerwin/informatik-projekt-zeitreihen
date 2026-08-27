"""Lädt die historischen Passantendaten aller Kartensensoren.

Aufruf aus dem Projektordner: ``python -m src.download_data``
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .data_config import DATA_END_UTC, DATA_START_UTC


BASE_URL = "https://smartes.bad-nauheim.de"
FEATURES_URL = (
    f"{BASE_URL}/Map/layers/57/features?"
    "filter=count%2CfeatureIdentifier%2Caverage%2Ctimestamp%2CtimeRangeStart"
)
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "sensordata.json"


def _parse_time(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


def _iso(value):
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _time_chunks(since, until):
    while since < until:
        end = min(since + timedelta(days=60), until)
        yield since, end
        since = end


def _get_json(url, attempts=3):
    """Wiederholt nur vorübergehende Netz- und Serverfehler."""
    request = Request(url, headers={"User-Agent": "AnomalieErkennung/1.0"})
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code < 500 or attempt == attempts - 1:
                raise
        except URLError:
            if attempt == attempts - 1:
                raise
        sleep(2 ** attempt)


def _deduplicate(records):
    unique = {}
    for record in records:
        key = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        unique[key] = record
    return sorted(unique.values(), key=lambda row: (
        row.get("timestamp", ""), row.get("SID", ""), row.get("name", ""),
        row.get("visitors", 0), row.get("avgDuration", 0),
    ))


def download_all(since, until, output=OUTPUT, fetch=_get_json):
    """Lädt alle eindeutigen Sensor-IDs in API-verträglichen Zeitblöcken."""
    since = _parse_time(since)
    until = _parse_time(until)
    output = Path(output)
    if since >= until:
        raise ValueError("--since muss vor --until liegen")

    feature_groups = fetch(FEATURES_URL)
    sensor_ids = sorted({
        int(feature["id"])
        for features in feature_groups.values()
        for feature in features
    })
    if not sensor_ids:
        raise ValueError("Die Karte lieferte keine Passantensensoren")

    records = []
    for number, sensor_id in enumerate(sensor_ids, 1):
        before = len(records)
        for start, end in _time_chunks(since, until):
            query = urlencode({"since": _iso(start), "until": _iso(end)})
            payload = fetch(f"{BASE_URL}/Map/history/{sensor_id}?{query}")
            records.extend(payload.get("sensordata", []))
        print(
            f"{number:02d}/{len(sensor_ids)} Sensor {sensor_id}: "
            f"{len(records) - before} Rohdatensätze"
        )

    # Die Stadt-API liefert den Endzeitpunkt mit. Für reproduzierbare,
    # halboffene Zeiträume wird die endgültige Grenze hier ausgeschlossen.
    records = [
        record for record in records
        if since <= _parse_time(record["timestamp"]) < until
    ]
    records = _deduplicate(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(
                {"sensordata": records}, file, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False,
            )
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"sensors": len(sensor_ids), "records": len(records), "output": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default=DATA_START_UTC)
    parser.add_argument("--until", default=DATA_END_UTC)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = download_all(args.since, args.until, args.output)
    print(
        f"Fertig: {result['records']} Datensätze von {result['sensors']} "
        f"Sensoren in {result['output']}"
    )


if __name__ == "__main__":
    main()
