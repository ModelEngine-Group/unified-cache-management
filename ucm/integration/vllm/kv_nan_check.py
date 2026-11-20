#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""Optional non-finite (NaN/Inf) probe for UCM KV transfers.

This module is a diagnostic probe, not a correctness mechanism: it never
modifies data and never fails a transfer. Enabling it only adds reads of the
KV blocks that a connector is about to dump, or has just loaded, and logs an
ERROR line when the bit patterns of a checked view contain non-finite values.

Environment variables
---------------------

``UCM_KV_NAN_CHECK``
    ``0``/unset (default) disables the probe. ``1``/``all`` checks both the
    dump and load phases. ``dump`` or ``load`` enables a single phase; a
    comma-separated list (``dump,load``) is also accepted. Unknown values
    disable the probe (fail-safe).

``UCM_KV_NAN_CHECK_GROUPS``
    Which store labels to check: a comma-separated subset of ``fa`` and
    ``wa`` (default ``fa,wa``). ``fa`` covers the full-attention compressed
    KV (the target KV); ``wa`` covers SWA / compressor tails.

``UCM_KV_NAN_CHECK_MAX_BLOCKS``
    Maximum blocks checked per group per call (default 64). Larger selections
    are evenly sampled (first and last block always kept).

``UCM_KV_NAN_CHECK_STRIDE``
    Check only every N-th probe opportunity (default 1).

Semantics and limitations
-------------------------

* Detection is by bit pattern, not ``torch.isnan``: bf16/fp16/fp32 exponent
  bits, and (for completeness on other hardware) fp8 e4m3/e5m2 codes.
* Granularity is the transferred token range, not the whole tensor block.
  Selections carry the per-row token offsets and per-row token counts the
  connector's pointer extraction computed, and only that sub-range of each
  block is checked. Regions outside the transfer are excluded: their
  content is not defined by UCM (stale KV or recycled pool bytes; an
  all-0xFF pool fill reads as NaN in bf16) and they are never copied into
  store records, so non-finite values there are expected and harmless. A
  selection without range metadata is checked whole-block.
* Integer and byte views are skipped entirely: int8/uint8 values cannot be
  non-finite. On A3, the DeepSeek-V4 indexer K cache is INT8 (skipped) while
  its FP16 scale view is checked as fp16.
* The probe only runs where tensor views exist (the worker role). Scheduler
  connectors have no layouts and are no-ops.
* Enabling the load phase forces synchronous loads: async loads complete
  after the step's forward has started, so a post-load read would race with
  compute and could report stale or torn data.
* The probe must never break serving: every internal failure is caught and
  logged as a warning.

This module imports only the standard library at import time; ``torch`` is
imported lazily inside the check paths so configuration and sampling logic
stay unit-testable without a torch runtime.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

__all__ = [
    "KVCheckSelection",
    "KVNaNCheckConfig",
    "KVNaNChecker",
    "sample_indices",
]

_PHASES = ("dump", "load")
_DEFAULT_GROUPS = frozenset({"fa", "wa"})
_OFF_TOKENS = frozenset({"", "0", "false", "no", "off", "none"})
_ALL_TOKENS = frozenset({"1", "true", "yes", "on", "all", "both"})

_MAX_LOG_ITEMS = 8


def _group_of(result: "_ViewResult") -> int:
    """Group token block size of a hit result (for range hints)."""

    return result.ranges[2] if result.ranges is not None else 0


def _parse_tokens(raw: Optional[str]) -> Optional[frozenset]:
    """Return lower-case tokens of a comma list, or None when unset."""

    if raw is None:
        return None
    return frozenset(token.strip().lower() for token in raw.split(",") if token.strip())


def _parse_int(raw: Optional[str], default: int, minimum: int) -> int:
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        return default
    return max(minimum, value)


def sample_indices(total: int, limit: int) -> list:
    """Evenly sample ``total`` indices, keeping first and last.

    ``total <= limit`` returns all indices. Otherwise ``limit`` positions are
    picked by even spacing and deduplicated, so the result length can be
    smaller than ``limit``. The first index (0) and the last (total - 1) are
    always included when ``limit >= 2``.
    """

    total = int(total)
    limit = max(1, int(limit))
    if total <= 0:
        return []
    if total <= limit:
        return list(range(total))
    if limit == 1:
        return [0]
    step = (total - 1) / (limit - 1)
    picks = {int(round(i * step)) for i in range(limit)}
    return sorted(picks)


@dataclass(frozen=True)
class KVNaNCheckConfig:
    """Parsed probe configuration; disabled by default."""

    phases: frozenset = frozenset()
    groups: frozenset = _DEFAULT_GROUPS
    max_blocks: int = 64
    stride: int = 1

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "KVNaNCheckConfig":
        source = os.environ if env is None else env
        raw = str(source.get("UCM_KV_NAN_CHECK", "")).strip().lower()
        if raw in _OFF_TOKENS:
            phases: frozenset = frozenset()
        elif raw in _ALL_TOKENS:
            phases = frozenset(_PHASES)
        else:
            tokens = _parse_tokens(raw)
            phases = frozenset(tokens & set(_PHASES)) if tokens else frozenset()

        group_tokens = _parse_tokens(source.get("UCM_KV_NAN_CHECK_GROUPS"))
        groups = (
            frozenset(group_tokens & set(_DEFAULT_GROUPS))
            if group_tokens is not None and (group_tokens & set(_DEFAULT_GROUPS))
            else _DEFAULT_GROUPS
        )
        return cls(
            phases=phases,
            groups=groups,
            max_blocks=_parse_int(source.get("UCM_KV_NAN_CHECK_MAX_BLOCKS"), 64, 1),
            stride=_parse_int(source.get("UCM_KV_NAN_CHECK_STRIDE"), 1, 1),
        )

    def phase_enabled(self, phase: str) -> bool:
        return phase in self.phases

    def group_enabled(self, label: str) -> bool:
        return str(label).lower() in self.groups


@dataclass
class KVCheckSelection:
    """One group's blocks picked for checking during a transfer.

    ``block_ids`` are physical vLLM block ids exactly as the connector's
    pointer extraction computed them (one entry per pointer row, so the same
    physical block may appear more than once); ``keys`` are the store keys
    for the same rows.

    ``token_offsets``/``token_counts`` describe the transferred token range
    of each row in **group token units** (the same values the connector's
    ``extract_addrs_with_offsets`` consumes): row ``i`` transfers tokens
    ``[token_offsets[i], token_offsets[i] + token_counts[i])`` of block
    ``block_ids[i]``, intersected with the block itself. ``None`` means the
    whole tensor block is transferred. ``token_block_size`` is the group's
    token block size used to scale the range into each view's own axis
    (``axis_offset = offset * view_axis_size // token_block_size`` — the
    same conversion as the pointer math). Rows whose clamped count is zero
    transfer nothing and are ignored.
    """

    group_id: int
    label: str
    keys: Optional[Sequence[Any]] = None
    block_ids: Optional[Sequence[Any]] = None
    token_offsets: Optional[Sequence[Any]] = None
    token_counts: Optional[Sequence[Any]] = None
    token_block_size: Optional[int] = None


@dataclass
class _ViewSpec:
    """Where a recorded block view keeps its token axis.

    ``axis_size`` is the view's token-axis block size in view units (may
    differ from the group's token block size, e.g. compressed states).
    ``whole_only`` marks views whose memory is not token-axis-sliceable:
    whole-page Ascend pages and standardized pages that interleave separate
    value/scale byte planes. Such views are checked whole-block, and rows
    whose transferred range is partial are skipped rather than mis-sliced.
    """

    layer_name: str
    tensor: Any
    token_axis: int
    axis_size: int
    whole_only: bool = False


def _parse_block_view(entry) -> Optional[_ViewSpec]:
    """Accept ``(layer, tensor)`` or ``(layer, tensor, axis, size, whole)``."""

    if not isinstance(entry, (tuple, list)) or len(entry) < 2:
        return None
    layer_name, tensor = entry[0], entry[1]
    token_axis, axis_size, whole_only = 1, -1, False
    if len(entry) >= 5:
        try:
            token_axis = int(entry[2])
            axis_size = int(entry[3])
            whole_only = bool(entry[4])
        except (TypeError, ValueError):
            return None
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.dim() <= token_axis:
        return None
    if axis_size < 0:
        axis_size = int(tensor.shape[token_axis])
    if axis_size <= 0:
        return None
    return _ViewSpec(
        layer_name=layer_name,
        tensor=tensor,
        token_axis=token_axis,
        axis_size=axis_size,
        whole_only=whole_only,
    )


@dataclass
class _ViewResult:
    """Reduction of one checked tensor view (no big tensors retained)."""

    selection: KVCheckSelection
    spec: _ViewSpec
    view_index: int
    sampled: list
    ids_tensor: Any
    ranges: Optional[tuple] = None
    any_bad: Any = None
    bad_count: Any = None

    @property
    def tensor(self) -> Any:
        return self.spec.tensor

    @property
    def layer_name(self) -> str:
        return self.spec.layer_name


def _rule_for(dtype: Any) -> Optional[tuple]:
    """Return ``(view_dtype, mask, threshold, mode)`` or None when skippable.

    Integer dtypes return None: their value range cannot encode non-finite
    numbers, so probing them would only add gather cost.
    """

    import torch

    if dtype == torch.bfloat16:
        return (torch.int16, 0x7FFF, 0x7F80, "ge")
    if dtype == torch.float16:
        return (torch.int16, 0x7FFF, 0x7C00, "ge")
    if dtype == torch.float32:
        return (torch.int32, 0x7FFFFFFF, 0x7F800000, "ge")
    e4m3 = getattr(torch, "float8_e4m3fn", None)
    if e4m3 is not None and dtype == e4m3:
        return (torch.uint8, 0x7F, 0x7F, "eq")
    e5m2 = getattr(torch, "float8_e5m2", None)
    if e5m2 is not None and dtype == e5m2:
        return (torch.uint8, 0x7F, 0x7C, "ge")
    return None


def _covers_whole(offsets: list, counts: list, group_block: int) -> bool:
    """True when the sampled rows transfer their whole tensor block."""

    return all(
        offset == 0 and count >= group_block for offset, count in zip(offsets, counts)
    )


def _nonfinite_mask(
    tensor: Any,
    ids_tensor: Any,
    rule: tuple,
    spec: Optional[_ViewSpec] = None,
    offsets: Optional[list] = None,
    counts: Optional[list] = None,
    group_block: Optional[int] = None,
) -> Any:
    """Gather rows and return a bool mask of non-finite bits.

    ``offsets``/``counts`` are the sampled rows' transferred ranges in group
    token units; ``None`` checks whole blocks. Ranges are converted per view
    with the same math as ``extract_addrs_with_offsets``:
    ``axis_offset = offset * axis_size // token_block_size``. Constant ranges
    narrow the gathered block before the bit test (less work); varying
    ranges AND a per-row position-validity mask into the full-block result.
    Returns ``None`` when nothing of this view is transferred.
    """

    import torch

    block = tensor.index_select(0, ids_tensor)

    if offsets is not None and spec is not None and group_block:
        axis = spec.axis_size
        view_off = [offset * axis // group_block for offset in offsets]
        view_cnt = [
            max(0, min(count * axis // group_block, axis - off)) if off < axis else 0
            for off, count in zip(view_off, counts)
        ]
        if set(view_cnt) == {0}:
            return None
        if len(set(zip(view_off, view_cnt))) == 1:
            start, length = view_off[0], view_cnt[0]
            if start == 0 and length >= axis:
                pass  # whole block: keep the gathered rows as-is
            elif length == 0:
                return None
            else:
                block = block.narrow(spec.token_axis, start, length)
        else:
            # Varying per-row ranges: full-block bit test, then AND a
            # per-row validity mask over token positions.
            starts = torch.as_tensor(view_off, device=block.device)
            ends = starts + torch.as_tensor(view_cnt, device=block.device)
            positions = torch.arange(axis, device=block.device)
            valid = (positions >= starts[:, None]) & (positions < ends[:, None])
            shape = [1] * block.dim()
            shape[0] = block.shape[0]
            shape[spec.token_axis] = axis
            return _apply_rule(block, rule) & valid.reshape(shape)

    return _apply_rule(block, rule)


def _apply_rule(block: Any, rule: tuple) -> Any:
    """Bit-pattern test for one gathered (possibly narrowed) block."""

    view_dtype, mask_value, threshold, mode = rule
    if block.dtype != view_dtype:
        block = block.view(view_dtype)
    bits = block & mask_value
    if mode == "eq":
        return bits == threshold
    return bits >= threshold


class KVNaNChecker:
    """Probe driver: gate by phase/stride, scan, aggregate, log, count."""

    def __init__(
        self,
        config: KVNaNCheckConfig,
        logger: Any = None,
        metrics: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.config = config
        self._logger = logger
        self._metrics = metrics
        self._call_counts: Dict[str, int] = {"dump": 0, "load": 0}

    @classmethod
    def from_env(
        cls,
        logger: Any = None,
        metrics: Optional[Callable[[dict], None]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> "KVNaNChecker":
        return cls(KVNaNCheckConfig.from_env(env), logger, metrics)

    @property
    def enabled(self) -> bool:
        return bool(self.config.phases)

    def phase_enabled(self, phase: str) -> bool:
        return self.config.phase_enabled(phase)

    def describe(self) -> str:
        config = self.config
        phases = ",".join(sorted(config.phases)) or "off"
        return (
            f"phases={phases}, groups={','.join(sorted(config.groups))}, "
            f"max_blocks={config.max_blocks}, stride={config.stride}"
        )

    def should_run(self, phase: str) -> bool:
        """Stride gate; counts every probe opportunity for the phase."""

        if phase not in self.config.phases:
            return False
        counter = self._call_counts.get(phase, 0)
        self._call_counts[phase] = counter + 1
        return counter % max(1, self.config.stride) == 0

    def run(self, selections, layouts, phase: str) -> None:
        """Scan selected blocks; never raises (probe must not break serving)."""

        try:
            self._run(selections, layouts, phase)
        except Exception as exc:  # pragma: no cover - defensive
            self._log(
                "warning",
                "UCM KV non-finite probe aborted (%s): %s",
                type(exc).__name__,
                exc,
            )

    def _run(self, selections, layouts, phase: str) -> None:
        if not self.should_run(phase):
            return
        if not selections:
            return
        import torch

        results = []
        skipped_partial = 0
        for selection in selections:
            if not self.config.group_enabled(selection.label):
                continue
            layout = (layouts or {}).get(selection.group_id)
            views = getattr(layout, "block_views", None)
            if not views:
                continue
            block_ids = selection.block_ids
            try:
                total = len(block_ids)
            except TypeError:
                continue
            if not total:
                continue
            sampled = sample_indices(total, self.config.max_blocks)
            offsets, counts, group_block = self._sampled_ranges(
                selection, sampled, total
            )
            ids_cache: dict = {}
            for view_index, entry in enumerate(views):
                spec = _parse_block_view(entry)
                if spec is None:
                    continue
                tensor = spec.tensor
                if tensor.shape[0] == 0:
                    continue
                rule = _rule_for(tensor.dtype)
                if rule is None:
                    continue
                if (
                    spec.whole_only
                    and offsets is not None
                    and not _covers_whole(offsets, counts, group_block)
                ):
                    # Whole-only views (whole pages, mixed value/scale planes)
                    # cannot express partial token ranges: check them only
                    # when the transfer covers the full page.
                    skipped_partial += 1
                    continue
                device_key = str(tensor.device)
                ids_tensor = ids_cache.get(device_key)
                if ids_tensor is None:
                    ids_tensor = torch.as_tensor(
                        sampled, dtype=torch.long, device=tensor.device
                    )
                    ids_cache[device_key] = ids_tensor
                mask = _nonfinite_mask(
                    tensor,
                    ids_tensor,
                    rule,
                    spec,
                    offsets,
                    counts,
                    group_block,
                )
                if mask is None:
                    continue
                results.append(
                    _ViewResult(
                        selection=selection,
                        spec=spec,
                        view_index=view_index,
                        sampled=sampled,
                        ids_tensor=ids_tensor,
                        ranges=(
                            None if offsets is None else (offsets, counts, group_block)
                        ),
                        any_bad=mask.any(),
                        bad_count=mask.sum(),
                    )
                )
        if skipped_partial:
            self._log(
                "debug",
                "UCM KV non-finite probe skipped %d whole-only views whose "
                "transferred range is partial.",
                skipped_partial,
            )
        if not results:
            return

        # One effective device sync: the first .item() drains the queue of
        # per-view reductions enqueued above on the current stream.
        hits = [result for result in results if bool(result.any_bad.item())]
        if not hits:
            return
        self._report(phase, hits, skipped_partial)

    def _sampled_ranges(
        self, selection: KVCheckSelection, sampled: list, total: int
    ) -> tuple:
        """Sampled transferred ranges in group token units, or Nones.

        Values are clamped to the block: offsets below zero become zero,
        ranges past the block end are truncated, and rows starting at or
        beyond the block transfer nothing.
        """

        offsets = selection.token_offsets
        counts = selection.token_counts
        group_block = selection.token_block_size
        if offsets is None or counts is None:
            return None, None, None
        try:
            group_block = int(group_block)
            full_off = [int(value) for value in offsets]
            full_cnt = [int(value) for value in counts]
        except (TypeError, ValueError):
            return None, None, None
        if group_block <= 0 or len(full_off) != total or len(full_cnt) != total:
            return None, None, None
        clamped = []
        for offset, count in zip(full_off, full_cnt):
            offset = max(0, offset)
            count = (
                max(0, min(count, group_block - offset)) if offset < group_block else 0
            )
            clamped.append((offset, count))
        offsets = [clamped[index][0] for index in sampled]
        counts = [clamped[index][1] for index in sampled]
        return offsets, counts, group_block

    def _report(self, phase: str, hits: list, skipped_partial: int = 0) -> None:
        details = []
        total_bad_elems = 0
        total_bad_blocks = 0
        for result in hits:
            bad_elems = int(result.bad_count.item())
            total_bad_elems += bad_elems
            bad_positions = self._bad_positions(result)
            total_bad_blocks += len(bad_positions)
            details.append(self._format_hit(result, bad_elems, bad_positions))

        header = (
            f"UCM KV non-finite probe hit: phase={phase} "
            f"views={len(hits)} bad_blocks={total_bad_blocks} "
            f"bad_elems={total_bad_elems}"
        )
        if skipped_partial:
            header += (
                f" (note: {skipped_partial} whole-only views with partial "
                "ranges were not checked)"
            )
        shown = details[:_MAX_LOG_ITEMS]
        hidden = len(details) - len(shown)
        message = "\n  ".join([header] + shown)
        if hidden > 0:
            message += f"\n  ... and {hidden} more views"
        self._log("error", message)

        if self._metrics is not None:
            try:
                self._metrics({"kv_nan_bad_blocks": float(total_bad_blocks)})
            except Exception:  # pragma: no cover - metrics must not break
                pass

    def _bad_positions(self, result: _ViewResult) -> list:
        """Second pass, only on hit views: positions of bad sampled blocks.

        Positions index into ``result.sampled``, which itself indexes the
        selection's full ``block_ids``/``keys`` lists, so they map to both the
        physical block id and the store key of the bad row.
        """

        try:
            rule = _rule_for(result.tensor.dtype)
            if rule is None:
                return []
            offsets, counts, group_block = (
                result.ranges if result.ranges is not None else (None, None, None)
            )
            mask = _nonfinite_mask(
                result.tensor,
                result.ids_tensor,
                rule,
                result.spec,
                offsets,
                counts,
                group_block,
            )
            if mask is None or mask.dim() == 0:
                return []
            per_block = mask.reshape(result.ids_tensor.numel(), -1).any(dim=1)
            return per_block.nonzero().flatten().tolist()
        except Exception as exc:  # pragma: no cover - defensive
            self._log(
                "warning",
                "UCM KV non-finite probe detail pass failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            return []

    def _format_hit(
        self, result: _ViewResult, bad_elems: int, bad_positions: list
    ) -> str:
        selection = result.selection
        bad_block_ids = [result.sampled[position] for position in bad_positions]
        bad_ids = ",".join(str(bid) for bid in bad_block_ids[:_MAX_LOG_ITEMS])
        if len(bad_block_ids) > _MAX_LOG_ITEMS:
            bad_ids += f"(+{len(bad_block_ids) - _MAX_LOG_ITEMS})"

        key_hint = "n/a"
        try:
            keys = selection.keys
            if keys and len(keys) == len(selection.block_ids):
                key_hint = ",".join(
                    bytes(keys[position])[:8].hex()
                    for position in bad_positions[:_MAX_LOG_ITEMS]
                )
                if len(bad_positions) > _MAX_LOG_ITEMS:
                    key_hint += f"(+{len(bad_positions) - _MAX_LOG_ITEMS})"
        except Exception:  # pragma: no cover - hint is best-effort
            key_hint = "n/a"

        range_hint = ""
        if result.ranges is not None:
            offsets, counts, _group_block = result.ranges
            if len(set(zip(offsets, counts))) == 1:
                range_hint = (
                    f" tokens={counts[0]}@{offsets[0]}"
                    if counts[0] < _group_of(result)
                    else ""
                )
            else:
                pairs = [
                    f"{offsets[position]}:{counts[position]}"
                    for position in bad_positions[:_MAX_LOG_ITEMS]
                ]
                range_hint = f" tokens=varied ranges={','.join(pairs) or '0'}"

        return (
            f"label={selection.label} group={selection.group_id} "
            f"view={result.view_index} layer={result.layer_name} "
            f"dtype={result.tensor.dtype} "
            f"checked={len(result.sampled)}/{len(selection.block_ids)} "
            f"bad_blocks=[{bad_ids}] bad_elems={bad_elems} "
            f"tokens_hint={range_hint.strip() or 'whole'} keys={key_hint}"
        )

    def _log(self, level: str, message: str, *args: Any) -> None:
        if args:
            try:
                message = message % args
            except Exception:  # pragma: no cover - formatting is ours
                pass
        target = self._logger
        if target is None:
            return
        try:
            log_method = getattr(target, level, None)
            if log_method is not None:
                log_method(message)
        except Exception:  # pragma: no cover - logging must not break
            pass
