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
    table_row_groups = [
        [_row_cells(row, include_bucket) for row in group]
        for group in _display_groups(visible_rows, include_bucket)
    ]
    table_rows = [row for group in table_row_groups for row in group]
    headers = (
        ["bucket", "metric", "values", "unit"]
        if include_bucket
        else [
            "metric",
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
    separator = "=" * len(lines[0])
    for group_index, group in enumerate(table_row_groups):
        lines.extend(
            "  ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
            for row in group
        )
        if include_bucket and group_index + 1 < len(table_row_groups):
            lines.append(separator)
    return "\n".join(lines)


def render_json(rows: list[QueryRow], limit: int | None = None) -> str:
    visible_rows = rows[:limit] if limit is not None else rows
    return json.dumps(
        [
            {
                "metric": row.metric,
                "bucket_start_ms": row.start_ms,
                "bucket_end_ms": row.end_ms,
                "values": _ordered_values(row.values),
                "unit": row.unit,
            }
            for row in visible_rows
        ],
        indent=2,
    )


def _display_groups(rows: list[QueryRow], include_bucket: bool) -> list[list[QueryRow]]:
    if not include_bucket:
        return [rows]
    groups: dict[str, list[QueryRow]] = {}
    for row in rows:
        groups.setdefault(row.metric, []).append(row)
    return [
        sorted(group, key=lambda row: (row.start_ms or 0, row.end_ms or 0))
        for group in groups.values()
    ]


def _row_cells(row: QueryRow, include_bucket: bool) -> list[str]:
    cells = []
    if include_bucket:
        cells.append(_format_bucket(row))
    cells.extend([row.metric, _format_values(row.values), row.unit])
    return cells


def _format_bucket(row: QueryRow) -> str:
    if row.start_ms is None or row.end_ms is None:
        return "-"
    return f"{_format_time(row.start_ms)}..{_format_time(row.end_ms)}"


def _format_time(timestamp_ms: int) -> str:
    return dt.datetime.fromtimestamp(timestamp_ms / 1000).strftime("%m-%d %H:%M:%S")


def _format_values(values: dict[str, float]) -> str:
    return " ".join(
        f"{key}={_format_number(value)}"
        for key, value in _ordered_values(values).items()
    )


def _ordered_values(values: dict[str, float]) -> dict[str, float]:
    preferred = ("p50", "p90", "p99", "avg")
    ordered = {key: values[key] for key in preferred if key in values}
    ordered.update(
        {key: value for key, value in values.items() if key not in preferred}
    )
    return ordered


def _format_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value and (abs_value >= 100000 or abs_value < 0.001):
        return f"{value:.3e}"
    return f"{value:.3f}"
