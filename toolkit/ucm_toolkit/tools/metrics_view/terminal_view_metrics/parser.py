from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSample:
    name: str
    labels: dict[str, str]
    value: float
    timestamp_ms: int | None = None


_SAMPLE_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+(-?\d+))?\s*$"
)


def parse_prometheus_text(text: str) -> list[MetricSample]:
    samples: list[MetricSample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if not match:
            continue
        name, labels_text, value_text, timestamp_text = match.groups()
        samples.append(
            MetricSample(
                name=name,
                labels=parse_labels(labels_text or ""),
                value=float(value_text),
                timestamp_ms=int(timestamp_text) if timestamp_text else None,
            )
        )
    return samples


def parse_labels(text: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    pos = 0
    length = len(text)
    while pos < length:
        while pos < length and text[pos] in " ,":
            pos += 1
        if pos >= length:
            break
        key_start = pos
        while pos < length and text[pos] != "=":
            pos += 1
        if pos >= length:
            break
        key = text[key_start:pos].strip()
        pos += 1
        if pos >= length or text[pos] != '"':
            break
        pos += 1
        chars: list[str] = []
        while pos < length:
            char = text[pos]
            pos += 1
            if char == "\\" and pos < length:
                escaped = text[pos]
                pos += 1
                if escaped == "n":
                    chars.append("\n")
                else:
                    chars.append(escaped)
                continue
            if char == '"':
                break
            chars.append(char)
        labels[key] = "".join(chars)
        while pos < length and text[pos] != ",":
            pos += 1
        if pos < length and text[pos] == ",":
            pos += 1
    return labels
