from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from .storage import MetricsStore


@dataclass(frozen=True)
class VectorPoint:
    labels: dict[str, str]
    value: float


Vector = dict[tuple[tuple[str, str], ...], VectorPoint]


def metric_names_in_expr(expr: str) -> set[str]:
    return set(
        re.findall(
            r"\b([A-Za-z_:][A-Za-z0-9_:]*)(?=\s*(?:\{|\[|$))",
            _clean_expr(expr),
        )
    )


class PromqlEvaluator:
    def __init__(
        self,
        store: MetricsStore,
        start_ms: int,
        end_ms: int,
        tag_filters: Mapping[str, str] | None = None,
        instant_counters: bool = False,
    ):
        self.store = store
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.tag_filters = tag_filters or {}
        self.instant_counters = instant_counters

    def evaluate(self, expr: str) -> Vector:
        return self._eval(_strip_outer_parens(_clean_expr(expr)))

    def _eval(self, expr: str) -> Vector:
        expr = _strip_outer_parens(expr.strip())
        if _is_number(expr):
            return {_key({}): VectorPoint({}, float(expr))}

        split = _split_set_operator(expr, "or")
        if split is not None:
            return _vector_or(self._eval(split[0]), self._eval(split[1]))

        call = _function_args(expr, "clamp_min")
        if call is not None:
            args = _split_top_level(call, ",")
            if len(args) != 2:
                raise ValueError(f"clamp_min expects 2 args: {expr}")
            return _vector_clamp_min(self._eval(args[0]), float(args[1]))

        call = _function_args(expr, "clamp_max")
        if call is not None:
            args = _split_top_level(call, ",")
            if len(args) != 2:
                raise ValueError(f"clamp_max expects 2 args: {expr}")
            return _vector_clamp_max(self._eval(args[0]), float(args[1]))

        call = _function_args(expr, "positive_or_nan")
        if call is not None:
            return _vector_positive_or_nan(self._eval(call))

        call = _function_args(expr, "histogram_quantile")
        if call is not None:
            args = _split_top_level(call, ",")
            if len(args) != 2:
                raise ValueError(f"histogram_quantile expects 2 args: {expr}")
            return self._histogram_quantile(float(args[0]), self._eval(args[1]))

        split = _split_binary(expr, ("+", "-"))
        if split is None:
            split = _split_binary(expr, ("*", "/"))
        if split is not None:
            left, op, right = split
            return _vector_binary(self._eval(left), self._eval(right), op)

        aggregate = _parse_sum_by(expr)
        if aggregate is not None:
            labels, inner = aggregate
            return _vector_sum_by(self._eval(inner), labels)

        for name in ("rate", "increase"):
            call = _function_args(expr, name)
            if call is not None:
                metric_name, matchers = _parse_selector(call)
                return self._counter_vector(metric_name, matchers, rate=name == "rate")

        metric_name, matchers = _parse_selector(expr)
        return self._gauge_vector(metric_name, matchers)

    def _counter_vector(
        self, metric_name: str, matchers: list[tuple[str, str, str]], rate: bool
    ) -> Vector:
        vector: Vector = {}
        for series in self.store.list_series(metric_name):
            labels = dict(series["labels"])
            if not _labels_match(labels, matchers) or not _tag_filters_match(
                labels, self.tag_filters
            ):
                continue
            samples = self.store.samples_for_series(
                int(series["id"]), self.start_ms, self.end_ms
            )
            if self.instant_counters:
                if not samples:
                    continue
                value = samples[-1][1]
            else:
                delta = _counter_delta(samples)
                if delta is None:
                    continue
                elapsed = max((samples[-1][0] - samples[0][0]) / 1000.0, 0.001)
                value = delta / elapsed if rate else delta
            vector[_key(labels)] = VectorPoint(labels, value)
        return vector

    def _gauge_vector(
        self, metric_name: str, matchers: list[tuple[str, str, str]]
    ) -> Vector:
        vector: Vector = {}
        for series in self.store.list_series(metric_name):
            labels = dict(series["labels"])
            if not _labels_match(labels, matchers) or not _tag_filters_match(
                labels, self.tag_filters
            ):
                continue
            samples = self.store.samples_for_series(
                int(series["id"]), self.start_ms, self.end_ms
            )
            if not samples:
                continue
            vector[_key(labels)] = VectorPoint(labels, samples[-1][1])
        return vector

    def _histogram_quantile(self, quantile: float, vector: Vector) -> Vector:
        buckets_by_group: dict[tuple[tuple[str, str], ...], dict[float, float]] = {}
        labels_by_group: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
        for point in vector.values():
            if "le" not in point.labels:
                continue
            labels = dict(point.labels)
            le = labels.pop("le")
            group = _key(labels)
            buckets_by_group.setdefault(group, {})[_parse_le(le)] = point.value
            labels_by_group[group] = labels

        result: Vector = {}
        for group, buckets in buckets_by_group.items():
            value = _histogram_quantile(quantile, buckets)
            if value is not None:
                result[group] = VectorPoint(labels_by_group[group], value)
        return result


def _clean_expr(expr: str) -> str:
    expr = expr.replace("$__rate_interval", "1m")
    expr = expr.replace("${perWorker:raw}", "worker_id")
    return re.sub(r"\s+", " ", expr).strip()


def _parse_selector(expr: str) -> tuple[str, list[tuple[str, str, str]]]:
    selector = re.sub(r"\[[^\]]+\]\s*$", "", expr.strip())
    match = re.fullmatch(r"([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?", selector)
    if not match:
        raise ValueError(f"Unsupported PromQL selector: {expr}")
    return match.group(1), _parse_matchers(match.group(2) or "")


def _parse_matchers(text: str) -> list[tuple[str, str, str]]:
    matchers: list[tuple[str, str, str]] = []
    for part in _split_top_level(text, ","):
        if not part.strip():
            continue
        match = re.fullmatch(
            r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*(=~|!~|=|!=)\s*"(.*)"\s*', part
        )
        if not match:
            continue
        value = bytes(match.group(3), "utf-8").decode("unicode_escape")
        matchers.append((match.group(1), match.group(2), value))
    return matchers


def _labels_match(
    labels: dict[str, str], matchers: Iterable[tuple[str, str, str]]
) -> bool:
    for name, op, value in matchers:
        if "$" in value:
            continue
        actual = labels.get(name, "")
        if op == "=" and actual != value:
            return False
        if op == "!=" and actual == value:
            return False
        if op == "=~" and re.fullmatch(value, actual) is None:
            return False
        if op == "!~" and re.fullmatch(value, actual) is not None:
            return False
    return True


def _tag_filters_match(labels: dict[str, str], filters: Mapping[str, str]) -> bool:
    return all(labels.get(name, "") == value for name, value in filters.items())


def _parse_sum_by(expr: str) -> tuple[list[str], str] | None:
    match = re.match(r"sum\s+by\s*\(([^)]*)\)\s*", expr)
    if not match:
        return None
    labels = [item.strip() for item in match.group(1).split(",") if item.strip()]
    rest = expr[match.end() :].strip()
    if not rest.startswith("(") or not rest.endswith(")"):
        return None
    return labels, rest[1:-1]


def _function_args(expr: str, name: str) -> str | None:
    prefix = f"{name}("
    if not expr.startswith(prefix) or not expr.endswith(")"):
        return None
    depth = 0
    for index, char in enumerate(expr[len(name) :], start=len(name)):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(expr) - 1:
                return None
    return expr[len(prefix) : -1]


def _split_binary(expr: str, operators: tuple[str, ...]) -> tuple[str, str, str] | None:
    depth = 0
    in_quote = False
    for index in range(len(expr) - 1, -1, -1):
        char = expr[index]
        if char == '"' and (index == 0 or expr[index - 1] != "\\"):
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == ")":
            depth += 1
            continue
        if char == "(":
            depth -= 1
            continue
        if depth == 0 and char in operators:
            if char == "-" and (index == 0 or expr[index - 1] in "+-*/("):
                continue
            return expr[:index].strip(), char, expr[index + 1 :].strip()
    return None


def _split_set_operator(expr: str, operator: str) -> tuple[str, str] | None:
    token = f" {operator} "
    depth = 0
    in_quote = False
    index = 0
    while index <= len(expr) - len(token):
        char = expr[index]
        if char == '"' and (index == 0 or expr[index - 1] != "\\"):
            in_quote = not in_quote
        elif not in_quote:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif depth == 0 and expr.startswith(token, index):
                return expr[:index], expr[index + len(token) :]
        index += 1
    return None


def _split_top_level(text: str, separator: str) -> list[str]:
    if not text:
        return []
    parts: list[str] = []
    depth = 0
    in_quote = False
    start = 0
    for index, char in enumerate(text):
        if char == '"' and (index == 0 or text[index - 1] != "\\"):
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def _strip_outer_parens(expr: str) -> str:
    while expr.startswith("(") and expr.endswith(")") and _balanced(expr[1:-1]):
        expr = expr[1:-1].strip()
    return expr


def _balanced(expr: str) -> bool:
    depth = 0
    in_quote = False
    for index, char in enumerate(expr):
        if char == '"' and (index == 0 or expr[index - 1] != "\\"):
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not in_quote


def _vector_sum_by(vector: Vector, labels: list[str]) -> Vector:
    sums: dict[tuple[tuple[str, str], ...], float] = {}
    group_labels: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    for point in vector.values():
        selected = {name: point.labels.get(name, "") for name in labels}
        key = _key(selected)
        sums[key] = sums.get(key, 0.0) + point.value
        group_labels[key] = selected
    return {key: VectorPoint(group_labels[key], value) for key, value in sums.items()}


def _vector_binary(left: Vector, right: Vector, op: str) -> Vector:
    left_scalar = _scalar_value(left)
    right_scalar = _scalar_value(right)
    result: Vector = {}
    if left_scalar is not None:
        for key, point in right.items():
            result[key] = VectorPoint(
                point.labels, _apply_binary(left_scalar, point.value, op)
            )
        return result
    if right_scalar is not None:
        for key, point in left.items():
            result[key] = VectorPoint(
                point.labels, _apply_binary(point.value, right_scalar, op)
            )
        return result
    for key in left.keys() & right.keys():
        result[key] = VectorPoint(
            left[key].labels, _apply_binary(left[key].value, right[key].value, op)
        )
    return result


def _vector_or(left: Vector, right: Vector) -> Vector:
    result = dict(right)
    result.update(left)
    return result


def _vector_clamp_min(vector: Vector, minimum: float) -> Vector:
    return {
        key: VectorPoint(point.labels, max(point.value, minimum))
        for key, point in vector.items()
    }


def _vector_clamp_max(vector: Vector, maximum: float) -> Vector:
    return {
        key: VectorPoint(point.labels, min(point.value, maximum))
        for key, point in vector.items()
    }


def _vector_positive_or_nan(vector: Vector) -> Vector:
    return {
        key: VectorPoint(point.labels, point.value if point.value > 0 else math.nan)
        for key, point in vector.items()
    }


def _scalar_value(vector: Vector) -> float | None:
    if set(vector.keys()) == {_key({})}:
        return next(iter(vector.values())).value
    return None


def _apply_binary(left: float, right: float, op: str) -> float:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return math.nan if right == 0 else left / right
    raise ValueError(f"Unsupported operator: {op}")


def _counter_delta(samples: list[tuple[int, float]]) -> float | None:
    if len(samples) < 2:
        return None
    total = 0.0
    previous = samples[0][1]
    for _, value in samples[1:]:
        total += value - previous if value >= previous else max(value, 0.0)
        previous = value
    return total


def _histogram_quantile(
    quantile: float, cumulative_buckets: dict[float, float]
) -> float | None:
    if not cumulative_buckets:
        return None
    buckets = sorted(cumulative_buckets.items(), key=lambda item: item[0])
    total = cumulative_buckets.get(math.inf, buckets[-1][1])
    if total <= 0:
        return None
    rank = quantile * total
    previous_bound: float | None = None
    previous_count = 0.0
    for upper_bound, cumulative_count in buckets:
        if cumulative_count < rank:
            previous_bound = upper_bound
            previous_count = cumulative_count
            continue
        if math.isinf(upper_bound):
            return previous_bound
        lower_bound = (
            0.0 if previous_bound is None and upper_bound > 0 else previous_bound
        )
        lower_bound = upper_bound if lower_bound is None else lower_bound
        bucket_count = cumulative_count - previous_count
        if bucket_count <= 0:
            return upper_bound
        return lower_bound + (upper_bound - lower_bound) * (
            (rank - previous_count) / bucket_count
        )
    finite_bounds = [bound for bound, _ in buckets if not math.isinf(bound)]
    return finite_bounds[-1] if finite_bounds else None


def _parse_le(value: str) -> float:
    return math.inf if value == "+Inf" else float(value)


def _key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False
