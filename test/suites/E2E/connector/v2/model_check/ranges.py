"""Compute expected transfer ranges directly from rank-local native KV views."""

from collections import defaultdict
from itertools import product
from reprlib import repr as short_repr

import numpy as np


def components(view):
    return view if isinstance(view, (tuple, list)) else (view,)


def fill_host_kv(views):
    """Fill each shared backing once, so misaddressed data is visible on readback."""
    import torch

    random = np.random.default_rng(1579)
    seen = set()
    for view in views.values():
        for tensor in components(view):
            storage = tensor.untyped_storage()
            if storage.data_ptr() in seen:
                continue
            seen.add(storage.data_ptr())
            raw = (
                torch.empty(0, dtype=torch.uint8)
                .set_(storage, 0, (storage.nbytes(),), (1,))
                .numpy()
            )
            for start in range(0, raw.size, 1 << 20):
                part = raw[start : start + (1 << 20)]
                part[:] = random.integers(0, 256, part.size, dtype=np.uint8)


def spans(tensor):
    """Enumerate the occupied byte intervals of a strided host tensor slice."""
    width = tensor.element_size()
    dims = sorted(
        (stride, size)
        for stride, size in zip(tensor.stride(), tensor.shape)
        if size > 1
    )
    base = tensor.data_ptr()
    outer = []
    for stride, size in dims:
        step = stride * tensor.element_size()
        if step == width:
            width *= size
        else:
            outer.append(range(0, step * size, step))
    for offsets in product(*outer):
        start = base + sum(offsets)
        yield start, start + width


def page_ranges(layer, view, block, first, stop, group, device_type):
    """Index native rows/entries first; byte offsets follow from PyTorch strides."""
    parts = components(view)
    if (
        device_type == "npu"
        and not group.is_state_snapshot
        and len(parts) == 1
        and parts[0].ndim == 5
    ):
        parts = parts[0].unbind(0)
    by_storage = defaultdict(list)
    for tensor in parts:
        if not tensor.numel():
            continue
        rows, remainder = divmod(tensor.size(0), layer.num_blocks)
        if remainder or rows < 1 or not 0 < block < layer.num_blocks:
            raise ValueError(
                f"Invalid native block {block} for {layer.layer_name}: {tensor.shape}"
            )
        page = tensor.narrow(0, block * rows, rows)
        dest = by_storage[tensor.untyped_storage().data_ptr()]
        if (first, stop) == (0, group.block_size):
            dest.extend(spans(page))
            continue
        axis = 2 if layer.attention_view_order == "BHNC" else 1
        per_row = page.size(axis)
        begin, r1 = divmod(first * rows * per_row, group.block_size)
        end, r2 = divmod(stop * rows * per_row, group.block_size)
        if r1 or r2:
            raise ValueError("Token window splits a native KV entry")
        for row in range(begin // per_row, (end - 1) // per_row + 1):
            lo = max(begin - row * per_row, 0)
            hi = min(end - row * per_row, per_row)
            dest.extend(spans(page[row].narrow(axis - 1, lo, hi - lo)))
    for intervals in by_storage.values():
        merged = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        yield from merged


def append_page(record, cursor, store, views, page, placement, include_local):
    # Distributed pages pack the distinct (TP,DCP) shards in order. Use
    # native tensor slices to measure each shard, including partial WA tails.
    block, first, stop = page
    slices = {}
    for index, (_, dcp_rank) in enumerate(placement.shards):
        lo = placement.local_position(first, dcp_rank)
        hi = placement.local_position(stop, dcp_rank)
        if lo == hi:
            continue
        if (lo, hi) not in slices:
            spans = [
                (begin, end - begin)
                for layer in store.group_layout.layer_specs
                for begin, end in page_ranges(
                    layer,
                    views[layer.layer_name],
                    block,
                    lo,
                    hi,
                    store.group,
                    store.spec.device_type,
                )
            ]
            slices[lo, hi] = spans, sum(size for _, size in spans)
        spans, width = slices[lo, hi]
        if index == placement.index and include_local:
            offset = cursor
            for ptr, size in spans:
                record.append((offset, ptr, size))
                offset += size
        cursor += width
    return cursor


def expected_records(
    layout,
    config,
    views,
    tables,
    placements,
    step_start,
    step_end,
    *,
    state_offloads=(),
    hit_tokens=0,
    state_load_target=None,
    only_state=False,
):
    """Index the native source or destination pages for each expected window."""
    from ucm.integration.vllm.v2.kv_cache_spec import RestorePolicy

    operation = "Dump" if state_load_target is None else "Get"
    if only_state:
        operation = "Get checkpoint"
    chunk = layout.spec.chunk_size
    state_ids = {
        gid
        for gid, layout_ in layout.store_layouts.items()
        if layout_.group.is_state_snapshot
    }
    checkpoints = defaultdict(dict)
    for gid, block, end in state_offloads:
        checkpoints[end][gid] = block
    expected_groups = {
        gid
        for gid in config.prefix_cacheable_group_ids
        if config.kv_cache_groups[gid].layer_names
    }
    if expected_groups != layout.store_layouts.keys() or len(tables) != len(
        config.kv_cache_groups
    ):
        raise AssertionError("Native and registered Worker groups disagree")
    records = defaultdict(list)
    global_sizes = defaultdict(int)
    for gid, store in layout.store_layouts.items():
        group = store.group
        placement = placements[gid]
        native = config.kv_cache_groups[gid]
        registered = {spec.layer_name for spec in store.group_layout.layer_specs}
        if (
            set(native.layer_names) != registered
            or group.block_token_span
            != native.kv_cache_spec.block_size * placement.dcp_size
        ):
            raise AssertionError(
                f"Worker disagrees with native layers or page size of group {gid}"
            )
        policy = group.restore_policy
        if policy is None or (only_state and policy != RestorePolicy.STATE):
            continue
        ends = range((step_start // chunk + 1) * chunk, step_end + 1, chunk)
        if policy in (RestorePolicy.WA, RestorePolicy.STATE):
            ends = ends[-1:]
        if policy == RestorePolicy.STATE and state_load_target is None:
            ends = sorted(
                end
                for end, blocks in checkpoints.items()
                if hit_tokens < end <= step_end
                and end % chunk == 0
                and state_ids <= blocks.keys()
            )
        for end in ends:
            start = end - chunk
            span = group.block_token_span
            if policy == RestorePolicy.STATE:
                block = (
                    checkpoints[end][gid]
                    if state_load_target is None
                    else tables[gid][(state_load_target - 1) // span]
                )
                pages = [(block, 0, span)]
            else:
                begin = end - group.tail_tokens if policy == RestorePolicy.WA else start
                pages = [
                    (
                        tables[gid][i],
                        max(0, begin - i * span),
                        min(span, end - i * span),
                    )
                    for i in range(begin // span, (end - 1) // span + 1)
                ]
            window = policy, start
            cursor = global_sizes[window]
            if cursor != layout.store_offsets_by_group[gid]:
                raise AssertionError(
                    f"Incorrect Store offset for {policy.name} group {gid}"
                )
            record = records[window]
            writer = placement.replicas[(start // chunk) % len(placement.replicas)]
            include_local = state_load_target is not None or writer == placement.rank
            for page in pages:
                cursor = append_page(
                    record, cursor, store, views, page, placement, include_local
                )
            global_sizes[window] = cursor
        print(
            f"{operation} group {gid}: {policy.name}, {len(ends)} checked windows, "
            f"{len(tables[gid])} block table entries {short_repr(tables[gid])}"
        )
    for (policy, start), size in global_sizes.items():
        if not size or size != layout.store_sizes_by_policy[policy]:
            raise AssertionError(
                f"Incomplete {policy.name} Store record at chunk {start}: {size} bytes"
            )
    return dict(records)
