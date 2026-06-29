from __future__ import annotations

import datetime as dt
import json

from .query import QueryRow


def render_table(rows: list[QueryRow], limit: int | None = None) -> str:
    visible_rows = rows[:limit] if limit is not None else rows
    if not visible_rows:
        return "No rows"
    include_bucket = any(
        row.start_ms is not None and row.end_ms is not None for row in visible_rows
    )
    table_rows = [_row_cells(row, include_bucket) for row in visible_rows]
    headers = (
        ["bucket", "metric", "group", "values", "unit"]
        if include_bucket
        else [
            "metric",
            "group",
            "values",
            "unit",
        ]
    )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in table_rows
    )
    return "\n".join(lines)


def render_json(rows: list[QueryRow], limit: int | None = None) -> str:
    visible_rows = rows[:limit] if limit is not None else rows
    return json.dumps(
        [
            {
                "metric": row.metric,
                "bucket_start_ms": row.start_ms,
                "bucket_end_ms": row.end_ms,
                "group": row.group,
                "values": row.values,
                "unit": row.unit,
            }
            for row in visible_rows
        ],
        indent=2,
        sort_keys=True,
    )


def _format_group(group: dict[str, str]) -> str:
    if not group:
        return "-"
    return ",".join(f"{key}={value}" for key, value in group.items())


def _row_cells(row: QueryRow, include_bucket: bool) -> list[str]:
    cells = []
    if include_bucket:
        cells.append(_format_bucket(row))
    cells.extend(
        [row.metric, _format_group(row.group), _format_values(row.values), row.unit]
    )
    return cells


def _format_bucket(row: QueryRow) -> str:
    if row.start_ms is None or row.end_ms is None:
        return "-"
    return f"{_format_time(row.start_ms)}..{_format_time(row.end_ms)}"


def _format_time(timestamp_ms: int) -> str:
    return dt.datetime.fromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def _format_values(values: dict[str, float]) -> str:
    return " ".join(f"{key}={_format_number(value)}" for key, value in values.items())


def _format_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value and (abs_value >= 100000 or abs_value < 0.001):
        return f"{value:.3e}"
    return f"{value:.3f}"
