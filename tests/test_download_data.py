"""Kleine Checks für den reproduzierbaren API-Abruf."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.download_data import _deduplicate, _time_chunks, download_all


class DownloadTests(unittest.TestCase):
    def test_chunks_are_contiguous_and_at_most_sixty_days(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 5, 2, tzinfo=timezone.utc)
        chunks = list(_time_chunks(start, end))
        self.assertEqual(chunks[0][0], start)
        self.assertEqual(chunks[-1][1], end)
        self.assertTrue(all(
            right - left <= timedelta(days=60) for left, right in chunks
        ))
        self.assertTrue(all(
            chunks[i][1] == chunks[i + 1][0]
            for i in range(len(chunks) - 1)
        ))

    def test_only_exact_duplicates_are_removed(self):
        first = {
            "SID": "a", "timestamp": "2025-01-01T00:00:00Z",
            "visitors": 1, "avgDuration": 2,
        }
        changed = dict(first, avgDuration=3)
        self.assertEqual(
            _deduplicate([first, dict(first), changed]), [first, changed]
        )

    def test_download_uses_unique_sensor_ids_and_atomic_output(self):
        calls = []
        record = {
            "SID": "a", "name": "Sensor A",
            "timestamp": "2025-01-01T00:00:00Z",
            "visitors": 1, "avgDuration": 2,
        }
        endpoint_record = dict(record, timestamp="2025-01-02T00:00:00Z")

        def fetch(url):
            calls.append(url)
            if "/features?" in url:
                return {"group": [{"id": 7}, {"id": 7}]}
            return {"sensordata": [record, dict(record), endpoint_record]}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw.json"
            result = download_all(
                "2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z",
                output, fetch,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result["sensors"], 1)
        self.assertEqual(result["records"], 1)
        self.assertEqual(payload["sensordata"], [record])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
