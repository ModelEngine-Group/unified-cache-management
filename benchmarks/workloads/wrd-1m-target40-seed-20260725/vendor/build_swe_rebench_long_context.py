#!/usr/bin/env python3
"""Build and validate the pinned 10k--128k SWE-rebench workload.

The builder keeps the trajectories from an existing reference profile and
uses its per-trajectory counts to bound the search while preserving the exact
global request count.  For every trajectory it selects exactly one contiguous
assistant-turn window, then performs the same deterministic W-way causal merge
used by the benchmark.  Prompt lengths in this file always mean the GLM-5.1
chat-template body; the trajectory-scoped salt is accounted for in a separate
``salted_prompt_tokens`` field.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
import math
import random
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEED = 20260725
ASSISTANT_PROMPT_MARKER = "<|assistant|></think>"
DEFAULT_PARQUET_SHA256 = (
    "14048dd1fcd22ce094b6e85f8a38f223a9ef1327031aaaad052804870212efa1"
)
DEFAULT_REFERENCE_PROFILES = Path(
    "/data/datasets/glm51-vllm-lmcache/processed/"
    "swe_rebench_openhands_trajectories/subset100_w32_noprimer/"
    "causal_w32_noprimer.seed-20260725.profiles.jsonl"
)
DEFAULT_PARQUET = Path(
    "/data/datasets/glm51-vllm-lmcache/raw/"
    "swe_rebench_openhands_trajectories/trajectories.parquet"
)
DEFAULT_TOOLS = Path(
    "/data/datasets/glm51-vllm-lmcache/raw/"
    "swe_rebench_openhands_trajectories/tools.json"
)
DEFAULT_TOKENIZER = Path("/data/models/GLM-5.1-FP8-12L")
DEFAULT_OUTPUT = Path(
    "/data/datasets/glm51-vllm-lmcache/processed/"
    "swe_rebench_openhands_trajectories/"
    "subset100_longctx10k_128k_mean40k_w32_noprimer_v1"
)
DEFAULT_PREFIX = "longctx10k_128k_mean40k_w32_noprimer.seed-20260725"
DEFAULT_SALT_LABEL = "swe-rebench-longctx40k-w32-noprimer-seed-20260725"


@dataclass(frozen=True)
class BuildConfig:
    min_tokens: int = 10_000
    max_tokens: int = 128_000
    fine_bin_tokens: int = 10_000
    target_mean_tokens: int = 40_000
    min_mean_tokens: int = 38_000
    max_mean_tokens: int = 42_000
    wide_bins: tuple[tuple[int, int], ...] = (
        (10_000, 20_000),
        (20_000, 40_000),
        (40_000, 60_000),
        (60_000, 100_000),
        (100_000, 128_000),
    )
    wide_targets: tuple[int, ...] = (1081, 1014, 676, 439, 169)
    expected_requests: int = 3379
    expected_trajectories: int = 100
    active_sessions: int = 32
    seed: int = SEED
    chunk_size: int = 256
    request_count_slack_fraction: float = 1.0
    request_count_slack_minimum: int = 0
    minimum_requests_per_trajectory: int = 5
    virtual_min_delay: float = 0.5
    virtual_median_delay: float = 1.0
    virtual_delay_sigma: float = 0.6

    @property
    def fine_bins(self) -> tuple[tuple[int, int], ...]:
        result: list[tuple[int, int]] = []
        lower = self.min_tokens
        while lower < self.max_tokens:
            upper = min(lower + self.fine_bin_tokens, self.max_tokens)
            result.append((lower, upper))
            lower = upper
        return tuple(result)

    def validate(self) -> None:
        if self.min_tokens >= self.max_tokens:
            raise ValueError("token interval must be increasing")
        if not self.min_mean_tokens <= self.target_mean_tokens <= self.max_mean_tokens:
            raise ValueError("target mean must be inside the accepted mean interval")
        if len(self.wide_bins) != len(self.wide_targets):
            raise ValueError("wide bins and targets must have equal length")
        if sum(self.wide_targets) != self.expected_requests:
            raise ValueError("wide targets must sum to expected requests")
        if self.wide_bins[0][0] != self.min_tokens:
            raise ValueError("wide bins must begin at min_tokens")
        if self.wide_bins[-1][1] != self.max_tokens:
            raise ValueError("wide bins must end at max_tokens")
        if any(
            left[1] != right[0]
            for left, right in zip(self.wide_bins, self.wide_bins[1:])
        ):
            raise ValueError("wide bins must be contiguous")
        if not 1 <= self.active_sessions <= self.expected_trajectories:
            raise ValueError("active_sessions must fit the trajectory count")
        if not 0 <= self.request_count_slack_fraction <= 1:
            raise ValueError("request-count slack fraction must be in [0, 1]")
        if self.request_count_slack_minimum < 0:
            raise ValueError("request-count minimum slack cannot be negative")
        if self.minimum_requests_per_trajectory < 1:
            raise ValueError("minimum requests per trajectory must be positive")


@dataclass(frozen=True)
class TrajectoryTurns:
    source_row: int
    trajectory_id: str
    instance_id: str
    repo: str
    resolved: int
    required_requests: int
    body_tokens: tuple[int, ...]
    salt_tokens: int


@dataclass(frozen=True)
class WindowCandidate:
    start: int
    stop: int
    token_sum: int
    wide_counts: tuple[int, ...]
    fine_counts: tuple[int, ...]


@dataclass(frozen=True)
class SelectionResult:
    candidates: tuple[WindowCandidate, ...]
    wide_counts: tuple[int, ...]
    fine_counts: tuple[int, ...]
    token_sum: int

    @property
    def mean_tokens(self) -> float:
        count = sum(self.wide_counts)
        return self.token_sum / count


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_jsonl_bytes(values: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(value) + b"\n" for value in values)


def manifest_canonical_sha256(manifest: dict[str, Any]) -> str:
    """Hash a manifest without its self-referential hash field."""
    canonical = copy.deepcopy(manifest)
    canonical.setdefault("integrity", {}).pop("manifest_canonical_sha256", None)
    return sha256_bytes(canonical_json_bytes(canonical))


def percentile(values: Sequence[int | float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def distribution(values: Iterable[int | float]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "mean": round(statistics.fmean(ordered), 3),
        "p50": round(percentile(ordered, 0.50), 3),
        "p95": round(percentile(ordered, 0.95), 3),
        "maximum": ordered[-1],
    }


def _bin_index(value: int, bins: Sequence[tuple[int, int]]) -> int:
    for index, (lower, upper) in enumerate(bins):
        if lower <= value < upper:
            return index
    raise ValueError(f"token length {value} is outside configured bins")


def histogram(
    values: Iterable[int], bins: Sequence[tuple[int, int]]
) -> tuple[int, ...]:
    counts = [0] * len(bins)
    for value in values:
        counts[_bin_index(value, bins)] += 1
    return tuple(counts)


def enumerate_windows(
    trajectory: TrajectoryTurns,
    config: BuildConfig,
    request_count: int | None = None,
) -> list[WindowCandidate]:
    lengths = trajectory.body_tokens
    if any(right <= left for left, right in zip(lengths, lengths[1:])):
        raise ValueError(
            f"body token ends are not increasing: {trajectory.trajectory_id}"
        )
    count = trajectory.required_requests if request_count is None else request_count
    if count <= 0 or count > len(lengths):
        raise ValueError(f"invalid required request count: {trajectory.trajectory_id}")
    result = []
    for start in range(len(lengths) - count + 1):
        stop = start + count
        selected = lengths[start:stop]
        if selected[0] < config.min_tokens or selected[-1] >= config.max_tokens:
            continue
        result.append(
            WindowCandidate(
                start=start,
                stop=stop,
                token_sum=sum(selected),
                wide_counts=histogram(selected, config.wide_bins),
                fine_counts=histogram(selected, config.fine_bins),
            )
        )
    if not result:
        raise ValueError(
            f"trajectory {trajectory.trajectory_id} has no contiguous "
            f"{count}-request window in [{config.min_tokens}, {config.max_tokens})"
        )
    return result


def enumerate_allowed_windows(
    trajectory: TrajectoryTurns,
    config: BuildConfig,
) -> list[WindowCandidate]:
    slack = max(
        config.request_count_slack_minimum,
        math.ceil(trajectory.required_requests * config.request_count_slack_fraction),
    )
    minimum = max(
        config.minimum_requests_per_trajectory,
        trajectory.required_requests - slack,
    )
    maximum = min(len(trajectory.body_tokens), trajectory.required_requests + slack)
    candidates = []
    for request_count in range(minimum, maximum + 1):
        try:
            candidates.extend(enumerate_windows(trajectory, config, request_count))
        except ValueError:
            # A particular count can be impossible inside the length interval
            # even when the trajectory's pinned reference count is feasible.
            continue
    if not candidates:
        raise ValueError(
            f"trajectory has no allowed window: {trajectory.trajectory_id}"
        )
    return candidates


def _selection_score(
    wide: Sequence[int],
    fine: Sequence[int],
    token_sum: int,
    config: BuildConfig,
) -> float:
    request_count = sum(wide)
    mean = token_sum / request_count
    missing_fine = sum(count == 0 for count in fine)
    mean_violation = (
        max(
            config.min_mean_tokens - mean,
            mean - config.max_mean_tokens,
            0.0,
        )
        / 1000.0
    )
    chi_square = sum(
        (actual - target) ** 2 / max(target, 1)
        for actual, target in zip(wide, config.wide_targets)
    )
    mean_delta = (mean - config.target_mean_tokens) / 1000.0
    return (
        missing_fine * 1e12 + mean_violation**2 * 1e9 + chi_square * 1e4 + mean_delta**2
    )


def _totals(chosen: Sequence[WindowCandidate]) -> tuple[list[int], list[int], int]:
    wide = [0] * len(chosen[0].wide_counts)
    fine = [0] * len(chosen[0].fine_counts)
    token_sum = 0
    for candidate in chosen:
        token_sum += candidate.token_sum
        for index, value in enumerate(candidate.wide_counts):
            wide[index] += value
        for index, value in enumerate(candidate.fine_counts):
            fine[index] += value
    return wide, fine, token_sum


def _solution_rank(
    selected: Sequence[WindowCandidate],
    wide: Sequence[int],
    fine: Sequence[int],
    token_sum: int,
    config: BuildConfig,
) -> tuple[Any, ...]:
    request_count = sum(wide)
    mean = token_sum / request_count
    return (
        abs(request_count - config.expected_requests),
        sum(count == 0 for count in fine),
        max(config.min_mean_tokens - mean, mean - config.max_mean_tokens, 0.0),
        sum(abs(value - target) for value, target in zip(wide, config.wide_targets)),
        sum((value - target) ** 2 for value, target in zip(wide, config.wide_targets)),
        abs(mean - config.target_mean_tokens),
        tuple((item.start, item.stop) for item in selected),
    )


def select_windows(
    trajectories: Sequence[TrajectoryTurns],
    config: BuildConfig,
    *,
    restarts: int = 32,
    coordinate_passes: int = 40,
    pair_passes: int = 16,
    anneal_steps: int = 600_000,
) -> SelectionResult:
    """Select one contiguous window per trajectory and keep the total exact."""
    config.validate()
    if len(trajectories) != config.expected_trajectories:
        raise ValueError(
            f"expected {config.expected_trajectories} trajectories, "
            f"found {len(trajectories)}"
        )
    if sum(item.required_requests for item in trajectories) != config.expected_requests:
        raise ValueError(
            "reference per-trajectory counts do not sum to expected requests"
        )
    choices = [enumerate_allowed_windows(item, config) for item in trajectories]
    choices_by_count = []
    for trajectory_choices in choices:
        by_count: dict[int, list[WindowCandidate]] = {}
        for candidate in trajectory_choices:
            by_count.setdefault(candidate.stop - candidate.start, []).append(candidate)
        choices_by_count.append(by_count)
    fixed_choices = [
        by_count[item.required_requests]
        for item, by_count in zip(trajectories, choices_by_count)
    ]
    rng = random.Random(config.seed)
    best: tuple[tuple[Any, ...], list[WindowCandidate]] | None = None

    for restart in range(max(restarts, 1)):
        order = list(range(len(trajectories)))
        rng.shuffle(order)
        chosen: list[WindowCandidate | None] = [None] * len(trajectories)
        wide = [0] * len(config.wide_bins)
        fine = [0] * len(config.fine_bins)
        token_sum = 0
        emitted = 0
        for trajectory_index in order:
            required = trajectories[trajectory_index].required_requests
            emitted_after = emitted + required
            fraction = emitted_after / config.expected_requests

            def partial_key(candidate: WindowCandidate) -> tuple[float, int]:
                projected_wide = [
                    wide[index] + candidate.wide_counts[index]
                    for index in range(len(wide))
                ]
                histogram_cost = sum(
                    (projected_wide[index] - config.wide_targets[index] * fraction) ** 2
                    / max(config.wide_targets[index] * fraction, 1.0)
                    for index in range(len(wide))
                )
                mean_cost = (
                    (
                        token_sum
                        + candidate.token_sum
                        - config.target_mean_tokens * emitted_after
                    )
                    / max(1000.0 * math.sqrt(emitted_after), 1.0)
                ) ** 2
                return histogram_cost + mean_cost, candidate.start

            ranked = sorted(fixed_choices[trajectory_index], key=partial_key)
            if restart == 0:
                candidate = ranked[0]
            else:
                candidate = ranked[rng.randrange(min(8, len(ranked)))]
            chosen[trajectory_index] = candidate
            emitted = emitted_after
            token_sum += candidate.token_sum
            for index, value in enumerate(candidate.wide_counts):
                wide[index] += value
            for index, value in enumerate(candidate.fine_counts):
                fine[index] += value

        concrete = [item for item in chosen if item is not None]
        wide, fine, token_sum = _totals(concrete)
        for _ in range(max(coordinate_passes, 1)):
            changed = False
            coordinate_order = list(range(len(trajectories)))
            rng.shuffle(coordinate_order)
            for trajectory_index in coordinate_order:
                current = concrete[trajectory_index]
                base_wide = [
                    value - current.wide_counts[index]
                    for index, value in enumerate(wide)
                ]
                base_fine = [
                    value - current.fine_counts[index]
                    for index, value in enumerate(fine)
                ]
                base_sum = token_sum - current.token_sum
                candidate = min(
                    choices_by_count[trajectory_index][current.stop - current.start],
                    key=lambda item: (
                        _selection_score(
                            [
                                base_wide[index] + item.wide_counts[index]
                                for index in range(len(base_wide))
                            ],
                            [
                                base_fine[index] + item.fine_counts[index]
                                for index in range(len(base_fine))
                            ],
                            base_sum + item.token_sum,
                            config,
                        ),
                        item.start,
                    ),
                )
                if candidate != current:
                    concrete[trajectory_index] = candidate
                    wide = [
                        base_wide[index] + candidate.wide_counts[index]
                        for index in range(len(base_wide))
                    ]
                    fine = [
                        base_fine[index] + candidate.fine_counts[index]
                        for index in range(len(base_fine))
                    ]
                    token_sum = base_sum + candidate.token_sum
                    changed = True
            if not changed:
                break

        rank = _solution_rank(concrete, wide, fine, token_sum, config)
        if best is None or rank < best[0]:
            best = rank, list(concrete)

    assert best is not None
    selected = best[1]

    # Explore bounded count changes in pairs.  A proposal gives one trajectory
    # exactly the number of requests removed from the other, so the global
    # request count is invariant rather than a soft optimization target.
    if anneal_steps > 0 and any(len(value) > 1 for value in choices_by_count):
        anneal_rng = random.Random(config.seed ^ 0x0A7731A1)
        current = list(selected)
        current_wide, current_fine, current_sum = _totals(current)
        variable_best = list(current)
        variable_best_totals = (list(current_wide), list(current_fine), current_sum)
        variable_best_rank = _solution_rank(
            current, current_wide, current_fine, current_sum, config
        )
        stages = 4
        steps_per_stage = max(1, anneal_steps // stages)
        for _ in range(stages):
            current = list(variable_best)
            current_wide, current_fine, current_sum = (
                list(variable_best_totals[0]),
                list(variable_best_totals[1]),
                variable_best_totals[2],
            )
            current_score = _selection_score(
                current_wide, current_fine, current_sum, config
            )
            for step in range(steps_per_stage):
                left_index, right_index = anneal_rng.sample(range(len(trajectories)), 2)
                left_current = current[left_index]
                right_current = current[right_index]
                left_new = anneal_rng.choice(choices[left_index])
                pair_count = (
                    left_current.stop
                    - left_current.start
                    + right_current.stop
                    - right_current.start
                )
                right_count = pair_count - (left_new.stop - left_new.start)
                right_options = choices_by_count[right_index].get(right_count)
                if not right_options:
                    continue
                right_new = anneal_rng.choice(right_options)
                proposal_wide = [
                    current_wide[index]
                    - left_current.wide_counts[index]
                    - right_current.wide_counts[index]
                    + left_new.wide_counts[index]
                    + right_new.wide_counts[index]
                    for index in range(len(current_wide))
                ]
                proposal_fine = [
                    current_fine[index]
                    - left_current.fine_counts[index]
                    - right_current.fine_counts[index]
                    + left_new.fine_counts[index]
                    + right_new.fine_counts[index]
                    for index in range(len(current_fine))
                ]
                proposal_sum = (
                    current_sum
                    - left_current.token_sum
                    - right_current.token_sum
                    + left_new.token_sum
                    + right_new.token_sum
                )
                proposal_score = _selection_score(
                    proposal_wide, proposal_fine, proposal_sum, config
                )
                fraction = step / max(steps_per_stage - 1, 1)
                temperature = 250_000.0 * (100.0 / 250_000.0) ** fraction
                delta = proposal_score - current_score
                if delta <= 0 or anneal_rng.random() < math.exp(
                    -min(delta / temperature, 700.0)
                ):
                    current[left_index] = left_new
                    current[right_index] = right_new
                    current_wide = proposal_wide
                    current_fine = proposal_fine
                    current_sum = proposal_sum
                    current_score = proposal_score
                    rank = _solution_rank(
                        current,
                        current_wide,
                        current_fine,
                        current_sum,
                        config,
                    )
                    if rank < variable_best_rank:
                        variable_best = list(current)
                        variable_best_totals = (
                            list(current_wide),
                            list(current_fine),
                            current_sum,
                        )
                        variable_best_rank = rank
        selected = variable_best

    # A one-trajectory move can be blocked when an earlier window improves the
    # 10--20k target but simultaneously loses one of the scarce 120--128k
    # samples.  Refine the best restart with deterministic two-trajectory
    # moves so those compensating shifts can happen atomically.
    wide, fine, token_sum = _totals(selected)
    pair_rng = random.Random(config.seed ^ 0x5A17C0DE)
    for _ in range(max(pair_passes, 0)):
        pair_order = list(range(len(trajectories)))
        pair_rng.shuffle(pair_order)
        for offset in range(0, len(pair_order) - 1, 2):
            left_index = pair_order[offset]
            right_index = pair_order[offset + 1]
            left_current = selected[left_index]
            right_current = selected[right_index]
            base_wide = [
                value
                - left_current.wide_counts[index]
                - right_current.wide_counts[index]
                for index, value in enumerate(wide)
            ]
            base_fine = [
                value
                - left_current.fine_counts[index]
                - right_current.fine_counts[index]
                for index, value in enumerate(fine)
            ]
            base_sum = token_sum - left_current.token_sum - right_current.token_sum

            def pair_key(
                pair: tuple[WindowCandidate, WindowCandidate],
            ) -> tuple[float, int, int]:
                left, right = pair
                return (
                    _selection_score(
                        [
                            base_wide[index]
                            + left.wide_counts[index]
                            + right.wide_counts[index]
                            for index in range(len(base_wide))
                        ],
                        [
                            base_fine[index]
                            + left.fine_counts[index]
                            + right.fine_counts[index]
                            for index in range(len(base_fine))
                        ],
                        base_sum + left.token_sum + right.token_sum,
                        config,
                    ),
                    left.start,
                    right.start,
                )

            current_pair = (left_current, right_current)
            left_options = choices_by_count[left_index][
                left_current.stop - left_current.start
            ]
            right_options = choices_by_count[right_index][
                right_current.stop - right_current.start
            ]
            best_pair = min(
                ((left, right) for left in left_options for right in right_options),
                key=pair_key,
            )
            if pair_key(best_pair) < pair_key(current_pair):
                selected[left_index], selected[right_index] = best_pair
                wide = [
                    base_wide[index]
                    + best_pair[0].wide_counts[index]
                    + best_pair[1].wide_counts[index]
                    for index in range(len(base_wide))
                ]
                fine = [
                    base_fine[index]
                    + best_pair[0].fine_counts[index]
                    + best_pair[1].fine_counts[index]
                    for index in range(len(base_fine))
                ]
                token_sum = base_sum + best_pair[0].token_sum + best_pair[1].token_sum

    wide, fine, token_sum = _totals(selected)
    result = SelectionResult(tuple(selected), tuple(wide), tuple(fine), token_sum)
    if sum(result.wide_counts) != config.expected_requests:
        raise RuntimeError("selection changed the exact global request count")
    if any(count == 0 for count in result.fine_counts):
        raise RuntimeError(f"selection misses fine token bins: {result.fine_counts}")
    if not config.min_mean_tokens <= result.mean_tokens <= config.max_mean_tokens:
        raise RuntimeError(
            f"selected mean {result.mean_tokens:.3f} is outside "
            f"[{config.min_mean_tokens}, {config.max_mean_tokens}]"
        )
    return result


def materialize_profiles(
    trajectories: Sequence[TrajectoryTurns],
    selection: SelectionResult,
    config: BuildConfig,
) -> list[dict[str, Any]]:
    profiles = []
    for trajectory, window in zip(trajectories, selection.candidates):
        steps = []
        previous: int | None = None
        for step_index, request_index in enumerate(range(window.start, window.stop)):
            body = trajectory.body_tokens[request_index]
            added = None if previous is None else body - previous
            steps.append(
                {
                    "step_index": step_index,
                    "request_index": request_index,
                    "phase": "measured",
                    "body_prompt_tokens": body,
                    "prompt_tokens": body,
                    "salt_tokens": trajectory.salt_tokens,
                    "salted_prompt_tokens": body + trajectory.salt_tokens,
                    "chunk_aligned_prompt_tokens": body
                    // config.chunk_size
                    * config.chunk_size,
                    "adjacent_added_body_tokens": added,
                }
            )
            previous = body
        profiles.append(
            {
                "source_row": trajectory.source_row,
                "trajectory_id": trajectory.trajectory_id,
                "instance_id": trajectory.instance_id,
                "repo": trajectory.repo,
                "resolved": trajectory.resolved,
                "assistant_calls": len(trajectory.body_tokens),
                "reference_measured_requests": trajectory.required_requests,
                "measured_requests": window.stop - window.start,
                "primer_request_index": None,
                "selected_window": {
                    "first_request_index": window.start,
                    "last_request_index": window.stop - 1,
                    "request_count": window.stop - window.start,
                    "reference_request_count": trajectory.required_requests,
                    "request_count_delta": (
                        window.stop - window.start - trajectory.required_requests
                    ),
                    "request_indices_contiguous": True,
                },
                "steps": steps,
            }
        )
    return profiles


def _sample_delay(rng: random.Random, config: BuildConfig) -> float:
    return config.virtual_min_delay + rng.lognormvariate(
        math.log(config.virtual_median_delay), config.virtual_delay_sigma
    )


def causal_merge(
    profiles: Sequence[dict[str, Any]], config: BuildConfig
) -> list[dict[str, Any]]:
    """Deterministically interleave sessions; W is not client concurrency."""
    rng = random.Random(config.seed)
    shuffled = list(profiles)
    rng.shuffle(shuffled)
    by_id = {str(item["trajectory_id"]): item for item in shuffled}
    queue: list[tuple[float, int, str, int]] = []
    activation = 0
    tie = 0
    active = 0

    def activate(base_time: float, initial: bool) -> None:
        nonlocal activation, tie, active
        if activation >= len(shuffled):
            return
        profile = shuffled[activation]
        activation += 1
        active += 1
        ready = rng.random() if initial else base_time + _sample_delay(rng, config)
        heapq.heappush(queue, (ready, tie, str(profile["trajectory_id"]), 0))
        tie += 1

    for _ in range(config.active_sessions):
        activate(0.0, True)

    events: list[dict[str, Any]] = []
    previous_request: dict[str, str] = {}
    while queue:
        ready, _, trajectory_id, step_index = heapq.heappop(queue)
        profile = by_id[trajectory_id]
        step = profile["steps"][step_index]
        request_id = f"{trajectory_id}:request-{step['request_index']}"
        events.append(
            {
                "ordinal": len(events),
                "request_id": request_id,
                "depends_on": previous_request.get(trajectory_id),
                "virtual_ready_time": round(ready, 9),
                "source_row": profile["source_row"],
                "trajectory_id": trajectory_id,
                "instance_id": profile["instance_id"],
                "repo": profile["repo"],
                "resolved": profile["resolved"],
                "active_sessions_at_emit": active,
                "unactivated_sessions_remaining": len(shuffled) - activation,
                **step,
            }
        )
        previous_request[trajectory_id] = request_id
        next_step = step_index + 1
        if next_step < len(profile["steps"]):
            heapq.heappush(
                queue,
                (ready + _sample_delay(rng, config), tie, trajectory_id, next_step),
            )
            tie += 1
        else:
            active -= 1
            activate(ready, False)
    return events


def validate_workload(
    profiles: Sequence[dict[str, Any]],
    order: Sequence[dict[str, Any]],
    config: BuildConfig,
) -> None:
    config.validate()
    if len(profiles) != config.expected_trajectories:
        raise ValueError("profile trajectory count mismatch")
    if len(order) != config.expected_requests:
        raise ValueError("order request count mismatch")
    expected_events: dict[str, dict[str, Any]] = {}
    all_body: list[int] = []
    for profile in profiles:
        steps = profile["steps"]
        if len(steps) != int(profile["measured_requests"]):
            raise ValueError("profile measured count mismatch")
        indices = [int(step["request_index"]) for step in steps]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError("profile request indices are not contiguous")
        for step in steps:
            if step["phase"] != "measured":
                raise ValueError("no-primer workload may only contain measured steps")
            body = int(step["body_prompt_tokens"])
            if not config.min_tokens <= body < config.max_tokens:
                raise ValueError("body token length outside configured interval")
            if int(step["prompt_tokens"]) != body:
                raise ValueError("legacy prompt_tokens must equal body_prompt_tokens")
            if int(step["salted_prompt_tokens"]) != body + int(step["salt_tokens"]):
                raise ValueError("salted prompt token accounting mismatch")
            request_id = f"{profile['trajectory_id']}:request-{step['request_index']}"
            expected_events[request_id] = step
            all_body.append(body)
    if len(expected_events) != config.expected_requests:
        raise ValueError("profile request IDs are not unique or count is wrong")
    if any(count == 0 for count in histogram(all_body, config.fine_bins)):
        raise ValueError("not every 10k-granularity token bin is covered")
    mean = statistics.fmean(all_body)
    if not config.min_mean_tokens <= mean <= config.max_mean_tokens:
        raise ValueError("body-token mean is outside accepted interval")

    seen: set[str] = set()
    previous_by_trajectory: dict[str, str] = {}
    seen_order_ids: set[str] = set()
    for ordinal, event in enumerate(order):
        if int(event["ordinal"]) != ordinal:
            raise ValueError("order ordinals are not contiguous")
        request_id = str(event["request_id"])
        if request_id not in expected_events or request_id in seen_order_ids:
            raise ValueError("order request set does not match profiles")
        dependency = event.get("depends_on")
        trajectory_id = str(event["trajectory_id"])
        if dependency != previous_by_trajectory.get(trajectory_id):
            raise ValueError("depends_on does not preserve trajectory order")
        if dependency is not None and dependency not in seen:
            raise ValueError("dependency points forward in the order")
        if not 1 <= int(event["active_sessions_at_emit"]) <= config.active_sessions:
            raise ValueError("active session count exceeds configured W")
        profile_step = expected_events[request_id]
        if int(event["body_prompt_tokens"]) != int(profile_step["body_prompt_tokens"]):
            raise ValueError("order/profile body token mismatch")
        previous_by_trajectory[trajectory_id] = request_id
        seen.add(request_id)
        seen_order_ids.add(request_id)
    if seen_order_ids != set(expected_events):
        raise ValueError("order omitted profile requests")


def _histogram_dict(
    counts: Sequence[int], bins: Sequence[tuple[int, int]]
) -> dict[str, int]:
    return {
        f"[{lower},{upper})": int(count) for (lower, upper), count in zip(bins, counts)
    }


def _reuse_summary(
    profiles: Sequence[dict[str, Any]], order: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    added: list[int] = []
    ratios: list[float] = []
    reusable = 0
    current_total = 0
    for profile in profiles:
        lengths = [int(step["body_prompt_tokens"]) for step in profile["steps"]]
        for previous, current in zip(lengths, lengths[1:]):
            added.append(current - previous)
            ratios.append(previous / current)
            reusable += previous
            current_total += current

    previous_ordinal: dict[str, int] = {}
    gap_requests: list[int] = []
    gap_sessions: list[int] = []
    gap_query_tokens: list[int] = []
    gap_distinct_tokens: list[int] = []
    for event in order:
        trajectory_id = str(event["trajectory_id"])
        prior = previous_ordinal.get(trajectory_id)
        if prior is not None:
            between = order[prior + 1 : int(event["ordinal"])]
            maximum_by_session: dict[str, int] = {}
            for other in between:
                other_id = str(other["trajectory_id"])
                maximum_by_session[other_id] = max(
                    maximum_by_session.get(other_id, 0),
                    int(other["body_prompt_tokens"]),
                )
            gap_requests.append(len(between))
            gap_sessions.append(len(maximum_by_session))
            gap_query_tokens.append(
                sum(int(other["body_prompt_tokens"]) for other in between)
            )
            gap_distinct_tokens.append(sum(maximum_by_session.values()))
        previous_ordinal[trajectory_id] = int(event["ordinal"])
    return {
        "adjacent_added_body_tokens": distribution(added),
        "reusable_prefix_ratio": distribution(ratios),
        "weighted_reusable_prefix_ratio": round(reusable / current_total, 9),
        "same_trajectory_intervening_requests": distribution(gap_requests),
        "same_trajectory_intervening_sessions": distribution(gap_sessions),
        "same_trajectory_intervening_query_tokens": distribution(gap_query_tokens),
        "same_trajectory_intervening_distinct_prefix_tokens": distribution(
            gap_distinct_tokens
        ),
    }


def build_manifest(
    profiles: Sequence[dict[str, Any]],
    order: Sequence[dict[str, Any]],
    config: BuildConfig,
    *,
    source: dict[str, Any],
    salt_label: str,
    prefix: str,
    profiles_bytes: bytes,
    order_bytes: bytes,
) -> dict[str, Any]:
    body = [
        int(step["body_prompt_tokens"])
        for profile in profiles
        for step in profile["steps"]
    ]
    salted = [
        int(step["salted_prompt_tokens"])
        for profile in profiles
        for step in profile["steps"]
    ]
    fine = histogram(body, config.fine_bins)
    wide = histogram(body, config.wide_bins)
    per_trajectory = []
    for profile in profiles:
        steps = profile["steps"]
        lengths = [int(step["body_prompt_tokens"]) for step in steps]
        per_trajectory.append(
            {
                "source_row": profile["source_row"],
                "trajectory_id": profile["trajectory_id"],
                "assistant_calls": profile["assistant_calls"],
                "request_count": len(steps),
                **profile["selected_window"],
                "body_prompt_tokens": distribution(lengths),
                "adjacent_added_body_tokens": distribution(
                    right - left for left, right in zip(lengths, lengths[1:])
                ),
                "reusable_prefix_ratio": distribution(
                    left / right for left, right in zip(lengths, lengths[1:])
                ),
            }
        )
    profiles_hash = sha256_bytes(profiles_bytes)
    order_hash = sha256_bytes(order_bytes)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "workload": {
            "requests": len(order),
            "trajectories": len(profiles),
            "active_session_pool": config.active_sessions,
            "active_session_pool_is_client_concurrency": False,
            "primer_included": False,
            "max_output_tokens": 1,
            "seed": config.seed,
            "prompt_salt_label": salt_label,
            "body_token_interval": {
                "minimum_inclusive": config.min_tokens,
                "maximum_exclusive": config.max_tokens,
            },
            "accepted_mean_interval": [
                config.min_mean_tokens,
                config.max_mean_tokens,
            ],
            "target_mean_tokens": config.target_mean_tokens,
            "trajectory_window_policy": {
                "one_contiguous_window_per_trajectory": True,
                "reference_count_slack_fraction": config.request_count_slack_fraction,
                "reference_count_slack_minimum": config.request_count_slack_minimum,
                "minimum_requests_per_trajectory": (
                    config.minimum_requests_per_trajectory
                ),
                "global_request_count_is_exact": True,
            },
        },
        "tokenization": {
            "assistant_prompt_marker": ASSISTANT_PROMPT_MARKER,
            "length_definition": (
                "GLM-5.1 chat-template body through the assistant generation "
                "marker; trajectory-scoped deterministic salt excluded"
            ),
            "salted_length_definition": "body_prompt_tokens + trajectory salt tokens",
            "enable_thinking": False,
            "chunk_size_tokens": config.chunk_size,
            "chat_template_sha256": source.get("chat_template_sha256"),
            "effective_chat_template_sha256": source.get(
                "effective_chat_template_sha256"
            ),
            "tokenizer_json_sha256": source.get("tokenizer_json_sha256"),
            "tools_sha256": source.get("tools_sha256"),
        },
        "length_distribution": {
            "body_prompt_tokens": distribution(body),
            "salted_prompt_tokens": distribution(salted),
            "fine_histogram": _histogram_dict(fine, config.fine_bins),
            "wide_histogram": {
                "target": _histogram_dict(config.wide_targets, config.wide_bins),
                "actual": _histogram_dict(wide, config.wide_bins),
                "delta": _histogram_dict(
                    [
                        value - target
                        for value, target in zip(wide, config.wide_targets)
                    ],
                    config.wide_bins,
                ),
            },
        },
        "causal_interleaving": {
            "algorithm": "seeded stochastic causal renewal-process merge",
            "seed": config.seed,
            "active_sessions": config.active_sessions,
            "primer_included": False,
            "depends_on_is_previous_selected_turn": True,
            "virtual_time_is_not_nominal_arrival_schedule": True,
            "virtual_delay": {
                "minimum": config.virtual_min_delay,
                "lognormal_median": config.virtual_median_delay,
                "lognormal_sigma": config.virtual_delay_sigma,
            },
        },
        "trajectory_window_summary": {
            "actual_request_counts": distribution(
                int(profile["measured_requests"]) for profile in profiles
            ),
            "reference_request_counts": distribution(
                int(profile["reference_measured_requests"]) for profile in profiles
            ),
            "request_count_delta": distribution(
                int(profile["measured_requests"])
                - int(profile["reference_measured_requests"])
                for profile in profiles
            ),
            "changed_trajectories": sum(
                int(profile["measured_requests"])
                != int(profile["reference_measured_requests"])
                for profile in profiles
            ),
            "all_windows_contiguous": True,
        },
        "reuse_and_intervening": _reuse_summary(profiles, order),
        "trajectories": per_trajectory,
        "source": source,
        "outputs": {
            "profiles": {
                "filename": f"{prefix}.profiles.jsonl",
                "canonical_sha256": profiles_hash,
                "size_bytes": len(profiles_bytes),
            },
            "order": {
                "filename": f"{prefix}.order.jsonl",
                "canonical_sha256": order_hash,
                "size_bytes": len(order_bytes),
            },
        },
        "integrity": {
            "canonicalization": "UTF-8, sorted keys, compact separators; JSONL has one LF per row",
            "profiles_canonical_sha256": profiles_hash,
            "order_canonical_sha256": order_hash,
        },
    }
    manifest["integrity"]["manifest_canonical_sha256"] = manifest_canonical_sha256(
        manifest
    )
    return manifest


def write_workload(
    output_directory: Path,
    prefix: str,
    profiles: Sequence[dict[str, Any]],
    order: Sequence[dict[str, Any]],
    config: BuildConfig,
    *,
    source: dict[str, Any],
    salt_label: str,
) -> dict[str, Any]:
    if not output_directory.is_absolute():
        raise ValueError("output directory must be absolute")
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite output directory: {output_directory}"
        )
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("output prefix must be a plain filename prefix")
    validate_workload(profiles, order, config)
    profiles_bytes = canonical_jsonl_bytes(profiles)
    order_bytes = canonical_jsonl_bytes(order)
    manifest = build_manifest(
        profiles,
        order,
        config,
        source=source,
        salt_label=salt_label,
        prefix=prefix,
        profiles_bytes=profiles_bytes,
        order_bytes=order_bytes,
    )
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir()
    (output_directory / f"{prefix}.profiles.jsonl").write_bytes(profiles_bytes)
    (output_directory / f"{prefix}.order.jsonl").write_bytes(order_bytes)
    (output_directory / f"{prefix}.manifest.json").write_bytes(manifest_bytes)
    return manifest


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message["role"])
    normalized: dict[str, Any] = {"role": role, "content": message.get("content") or ""}
    if role == "assistant" and message.get("tool_calls"):
        calls = []
        for call in message["tool_calls"]:
            function = call["function"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw_arguments": arguments}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            calls.append(
                {
                    "id": call.get("id"),
                    "type": call.get("type") or "function",
                    "function": {"name": function["name"], "arguments": arguments},
                }
            )
        normalized["tool_calls"] = calls
    if role == "tool":
        normalized["name"] = message.get("name")
        normalized["tool_call_id"] = message.get("tool_call_id")
    return normalized


def subsequence_ends(values: Sequence[int], marker: Sequence[int]) -> list[int]:
    return [
        start + len(marker)
        for start in range(len(values) - len(marker) + 1)
        if list(values[start : start + len(marker)]) == list(marker)
    ]


def _load_selected_rows(
    parquet_path: Path, source_rows: set[int]
) -> dict[int, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError("pyarrow is required to read SWE-rebench Parquet") from error
    parquet = pq.ParquetFile(parquet_path)
    selected: dict[int, dict[str, Any]] = {}
    global_start = 0
    columns = ["trajectory_id", "instance_id", "repo", "trajectory", "resolved"]
    for row_group in range(parquet.metadata.num_row_groups):
        row_count = parquet.metadata.row_group(row_group).num_rows
        group_end = global_start + row_count
        if any(global_start <= row < group_end for row in source_rows):
            offset = 0
            for batch in parquet.iter_batches(
                batch_size=32, columns=columns, row_groups=[row_group]
            ):
                for row in batch.to_pylist():
                    source_row = global_start + offset
                    offset += 1
                    if source_row in source_rows:
                        selected[source_row] = {**row, "source_row": source_row}
        global_start = group_end
    if set(selected) != source_rows:
        raise RuntimeError("not all reference source rows were loaded from Parquet")
    return selected


def tokenize_trajectories(
    reference_profiles: Sequence[dict[str, Any]],
    parquet_path: Path,
    tools_path: Path,
    tokenizer_path: Path,
    salt_label: str,
) -> tuple[list[TrajectoryTurns], dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "transformers is required for GLM-5.1 tokenization"
        ) from error
    selected_rows = _load_selected_rows(
        parquet_path, {int(profile["source_row"]) for profile in reference_profiles}
    )
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    marker_ids = tokenizer.encode(ASSISTANT_PROMPT_MARKER, add_special_tokens=False)
    trajectories = []
    for number, reference in enumerate(reference_profiles, start=1):
        source_row = int(reference["source_row"])
        row = selected_rows[source_row]
        if str(row["trajectory_id"]) != str(reference["trajectory_id"]):
            raise ValueError(f"trajectory ID mismatch at source row {source_row}")
        messages = [normalize_message(message) for message in row["trajectory"]]
        tokenized = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        full_ids = tokenized["input_ids"] if hasattr(tokenized, "keys") else tokenized
        if hasattr(full_ids, "tolist"):
            full_ids = full_ids.tolist()
        lengths = subsequence_ends(full_ids, marker_ids)
        assistant_calls = sum(message["role"] == "assistant" for message in messages)
        if len(lengths) != assistant_calls:
            raise RuntimeError(
                f"assistant marker mismatch at source row {source_row}: "
                f"{len(lengths)} != {assistant_calls}"
            )
        salt = (
            f"<|workload_namespace|>{salt_label}|"
            f"trajectory={reference['trajectory_id']}\n"
        )
        trajectories.append(
            TrajectoryTurns(
                source_row=source_row,
                trajectory_id=str(reference["trajectory_id"]),
                instance_id=str(reference["instance_id"]),
                repo=str(reference["repo"]),
                resolved=int(reference["resolved"]),
                required_requests=int(reference["measured_requests"]),
                body_tokens=tuple(int(value) for value in lengths),
                salt_tokens=len(tokenizer.encode(salt, add_special_tokens=False)),
            )
        )
        if number % 10 == 0:
            print(
                f"tokenized trajectories={number}/{len(reference_profiles)}",
                file=sys.stderr,
            )
    chat_template = tokenizer.chat_template or tokenizer.default_chat_template
    chat_template_path = tokenizer_path / "chat_template.jinja"
    source_tokenization = {
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_json_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
        "chat_template": str(chat_template_path.resolve()),
        "chat_template_sha256": sha256_file(chat_template_path),
        "effective_chat_template_sha256": sha256_bytes(
            str(chat_template).encode("utf-8")
        ),
        "tools_sha256": sha256_file(tools_path),
    }
    return trajectories, source_tokenization


def validate_existing(directory: Path, config: BuildConfig) -> dict[str, Any]:
    if not directory.is_absolute():
        raise ValueError("validation directory must be absolute")
    manifests = list(directory.glob("*.manifest.json"))
    if len(manifests) != 1:
        raise ValueError("validation directory must contain exactly one manifest")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    profiles_path = directory / manifest["outputs"]["profiles"]["filename"]
    order_path = directory / manifest["outputs"]["order"]["filename"]
    profiles = load_jsonl(profiles_path)
    order = load_jsonl(order_path)
    validate_workload(profiles, order, config)
    if sha256_file(profiles_path) != manifest["integrity"]["profiles_canonical_sha256"]:
        raise ValueError("profile canonical hash mismatch")
    if sha256_file(order_path) != manifest["integrity"]["order_canonical_sha256"]:
        raise ValueError("order canonical hash mismatch")
    expected_manifest_hash = manifest["integrity"]["manifest_canonical_sha256"]
    if manifest_canonical_sha256(manifest) != expected_manifest_hash:
        raise ValueError("manifest canonical hash mismatch")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-profiles", type=Path, default=DEFAULT_REFERENCE_PROFILES
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--prompt-salt-label", default=DEFAULT_SALT_LABEL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--active-sessions", type=int, default=32)
    parser.add_argument("--request-count-slack-fraction", type=float, default=1.0)
    parser.add_argument("--request-count-slack-minimum", type=int, default=0)
    parser.add_argument("--minimum-requests-per-trajectory", type=int, default=5)
    parser.add_argument("--search-restarts", type=int, default=32)
    parser.add_argument("--coordinate-passes", type=int, default=40)
    parser.add_argument("--pair-passes", type=int, default=16)
    parser.add_argument("--anneal-steps", type=int, default=600_000)
    parser.add_argument("--expected-parquet-sha256", default=DEFAULT_PARQUET_SHA256)
    parser.add_argument("--validate-existing", type=Path)
    return parser.parse_args()


def _require_absolute(paths: Iterable[tuple[str, Path]]) -> None:
    for name, path in paths:
        if not path.is_absolute():
            raise ValueError(f"{name} must be an absolute path: {path}")


def main() -> int:
    args = parse_args()
    config = BuildConfig(
        seed=args.seed,
        active_sessions=args.active_sessions,
        request_count_slack_fraction=args.request_count_slack_fraction,
        request_count_slack_minimum=args.request_count_slack_minimum,
        minimum_requests_per_trajectory=args.minimum_requests_per_trajectory,
    )
    config.validate()
    if args.validate_existing is not None:
        manifest = validate_existing(args.validate_existing, config)
        print(
            json.dumps(
                {"status": "valid", "integrity": manifest["integrity"]}, indent=2
            )
        )
        return 0
    _require_absolute(
        (
            ("--reference-profiles", args.reference_profiles),
            ("--parquet", args.parquet),
            ("--tools", args.tools),
            ("--tokenizer", args.tokenizer),
            ("--output-directory", args.output_directory),
        )
    )
    parquet_hash = sha256_file(args.parquet)
    if parquet_hash != args.expected_parquet_sha256:
        raise RuntimeError(
            f"unexpected Parquet SHA-256: {parquet_hash} != {args.expected_parquet_sha256}"
        )
    reference_profiles = load_jsonl(args.reference_profiles)
    trajectories, tokenization_source = tokenize_trajectories(
        reference_profiles,
        args.parquet,
        args.tools,
        args.tokenizer,
        args.prompt_salt_label,
    )
    selection = select_windows(
        trajectories,
        config,
        restarts=args.search_restarts,
        coordinate_passes=args.coordinate_passes,
        pair_passes=args.pair_passes,
        anneal_steps=args.anneal_steps,
    )
    profiles = materialize_profiles(trajectories, selection, config)
    order = causal_merge(profiles, config)
    source = {
        "reference_profiles": {
            "path": str(args.reference_profiles.resolve()),
            "sha256": sha256_file(args.reference_profiles),
        },
        "parquet": {"path": str(args.parquet.resolve()), "sha256": parquet_hash},
        "tools": {"path": str(args.tools.resolve()), "sha256": sha256_file(args.tools)},
        **tokenization_source,
    }
    manifest = write_workload(
        args.output_directory,
        args.output_prefix,
        profiles,
        order,
        config,
        source=source,
        salt_label=args.prompt_salt_label,
    )
    print(
        json.dumps(
            {
                "output_directory": str(args.output_directory),
                "body_prompt_tokens": manifest["length_distribution"][
                    "body_prompt_tokens"
                ],
                "wide_histogram": manifest["length_distribution"]["wide_histogram"],
                "integrity": manifest["integrity"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
