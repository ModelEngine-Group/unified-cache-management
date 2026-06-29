from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Iterable

from .parser import MetricSample


class MetricsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MetricsStore":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
CREATE TABLE IF NOT EXISTS series (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    labels_json TEXT NOT NULL,
    labels_hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_series_name ON series(name);
CREATE TABLE IF NOT EXISTS samples (
    series_id INTEGER NOT NULL,
    ts_ms INTEGER NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY(series_id, ts_ms),
    FOREIGN KEY(series_id) REFERENCES series(id)
);
CREATE INDEX IF NOT EXISTS idx_samples_series_ts ON samples(series_id, ts_ms);
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""
        )
        self.conn.commit()

    def write_samples(
        self,
        samples: Iterable[MetricSample],
        scrape_ts_ms: int,
        include_names: set[str] | None = None,
    ) -> int:
        rows: list[tuple[int, int, float]] = []
        for sample in samples:
            if include_names is not None and sample.name not in include_names:
                continue
            if not math.isfinite(sample.value):
                continue
            series_id = self._series_id(sample.name, sample.labels)
            rows.append(
                (
                    series_id,
                    (
                        sample.timestamp_ms
                        if sample.timestamp_ms is not None
                        else scrape_ts_ms
                    ),
                    sample.value,
                )
            )
        if not rows:
            return 0
        self.conn.executemany(
            "INSERT OR REPLACE INTO samples(series_id, ts_ms, value) VALUES (?, ?, ?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def _series_id(self, name: str, labels: dict[str, str]) -> int:
        labels_json = json.dumps(labels, sort_keys=True, separators=(",", ":"))
        labels_hash = hashlib.sha1(f"{name}\0{labels_json}".encode()).hexdigest()
        row = self.conn.execute(
            "SELECT id FROM series WHERE labels_hash = ?", (labels_hash,)
        ).fetchone()
        if row:
            return int(row[0])
        cursor = self.conn.execute(
            "INSERT INTO series(name, labels_json, labels_hash) VALUES (?, ?, ?)",
            (name, labels_json, labels_hash),
        )
        return int(cursor.lastrowid)

    def list_series(self, name: str) -> list[dict[str, object]]:
        rows = self.conn.execute(
            "SELECT id, name, labels_json FROM series WHERE name = ? ORDER BY id",
            (name,),
        ).fetchall()
        return [
            {"id": int(row[0]), "name": row[1], "labels": json.loads(row[2])}
            for row in rows
        ]

    def samples_for_series(
        self, series_id: int, start_ms: int, end_ms: int
    ) -> list[tuple[int, float]]:
        return [
            (int(row[0]), float(row[1]))
            for row in self.conn.execute(
                """
SELECT ts_ms, value FROM samples
WHERE series_id = ? AND ts_ms >= ? AND ts_ms <= ?
ORDER BY ts_ms
""",
                (series_id, start_ms, end_ms),
            ).fetchall()
        ]

    def latest_ts_ms(self) -> int | None:
        row = self.conn.execute("SELECT MAX(ts_ms) FROM samples").fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def clear(self) -> None:
        self.conn.executescript(
            """
DELETE FROM samples;
DELETE FROM series;
DELETE FROM metadata;
DELETE FROM sqlite_sequence WHERE name = 'series';
"""
        )
        self.conn.commit()
        self.conn.execute("VACUUM")

    def prune_before(self, cutoff_ms: int) -> int:
        cursor = self.conn.execute("DELETE FROM samples WHERE ts_ms < ?", (cutoff_ms,))
        self.conn.commit()
        return int(cursor.rowcount)
