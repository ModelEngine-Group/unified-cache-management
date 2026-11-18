"""Export FAWA byte layouts and inspect completed, uncompressed Posix files.

Enable startup export with UCM_KV_CACHE_METADATA_DIR. Each worker writes one
JSON per FA/WA store after its Store initializes.
The command line uses only the Python standard library::

    python ucm/integration/vllm/kv_cache_metadata.py \
        --metadata /tmp/kv-meta/worker-FA.json /cache/fawa_fa/data/<key>

Registered byte views remain opaque unless an overlapping typed alias describes
their contents. Padding and opaque bytes are reported separately. This checks
record size and floating-point encodings, not correctness of finite KV values.
"""

import argparse
import itertools
import json
import os
import socket
import struct
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Unsigned bit representation, sign mask, largest non-NaN encoding, infinity.
FLOAT_FORMATS = {
    "torch.bfloat16": ("H", 0x7FFF, 0x7F80, 0x7F80),
    "torch.float16": ("H", 0x7FFF, 0x7C00, 0x7C00),
    "torch.float32": ("I", 0x7FFFFFFF, 0x7F800000, 0x7F800000),
    "torch.float64": ("Q", 0x7FFFFFFFFFFFFFFF, 0x7FF0000000000000, 0x7FF0000000000000),
    "torch.float8_e4m3fn": ("B", 0x7F, 0x7E, None),
    "torch.float8_e5m2": ("B", 0x7F, 0x7C, 0x7C),
    "torch.float8_e4m3fnuz": ("B", 0xFF, None, None),
    "torch.float8_e5m2fnuz": ("B", 0xFF, None, None),
    "torch.float8_e8m0fnu": ("B", 0xFF, 0xFE, None),
}
INTEGER_DTYPES = {
    "torch.int8",
    "torch.int16",
    "torch.int32",
    "torch.int64",
    "torch.bool",
}
SPEC_FIELDS = (
    "block_size",
    "storage_block_size",
    "tokens_per_state",
    "dtype",
    "head_size",
    "num_kv_heads",
    "scale_dim",
    "scale_dtype",
    "cache_dtype_str",
    "model_version",
    "state_content_size_bytes",
    "page_size_bytes",
    "unpadded_page_size_bytes",
    "page_size_padded",
    "sliding_window",
    "prefix_cacheable",
    "prefix_replay_tokens",
    "kv_quant_mode",
    "cache_sparse_sfa_c8",
    "cache_sparse_sfa_turboquant",
    "cache_sparse_li_c8",
    "cache_sparse_li_c4",
    "is_quantized",
)


def _spec_metadata(spec):
    result = {"type": f"{type(spec).__module__}.{type(spec).__qualname__}"}
    for field in SPEC_FIELDS:
        value = getattr(spec, field, None)
        if value is not None:
            result[field] = (
                value if isinstance(value, (bool, int, float, str)) else str(value)
            )
    return result


def _views(kv_caches, group):
    """Keep typed aliases even when the transfer uses a whole-page byte view."""
    result = []
    nested = getattr(group.kv_cache_spec, "kv_cache_specs", None)
    for name in group.layer_names:
        value = kv_caches[name]
        tensors = value if isinstance(value, (tuple, list)) else (value,)
        spec = nested[name] if nested else group.kv_cache_spec
        for component, tensor in enumerate(tensors):
            blocks = (
                (tensor[0], tensor[1])
                if tensor.dim() == 5 and tensor.shape[0] == 2
                else (tensor,)
            )
            for part, block_view in enumerate(blocks):
                width = int(block_view.element_size())
                dtype = str(block_view.dtype)
                packed_float8 = "float8" in dtype and (
                    getattr(spec, "model_version", None)
                    in ("deepseek_v4", "deepseek_v41")
                    or getattr(spec, "cache_dtype_str", None)
                    in ("fp8_ds_mla", "nvfp4_ds_mla")
                )
                result.append(
                    {
                        "name": name,
                        "component": f"{component}.{part}",
                        "dtype": dtype,
                        "element_bytes": width,
                        "shape": list(block_view.shape),
                        "stride_bytes": [int(n) * width for n in block_view.stride()],
                        "storage_offset_bytes": int(block_view.storage_offset())
                        * width,
                        "storage_id": f"{block_view.device}:{block_view.untyped_storage().data_ptr()}",
                        "address": int(block_view.data_ptr()),
                        "numeric": not packed_float8
                        and dtype in (FLOAT_FORMATS.keys() | INTEGER_DTYPES),
                    }
                )
    return result


def _view_ranges(view):
    """Yield dense runs in block zero, preserving holes in strided views."""
    if not all(view["shape"]):
        return
    dimensions = sorted(
        (stride, size)
        for stride, size in zip(view["stride_bytes"][1:], view["shape"][1:])
        if size > 1
    )
    length = view["element_bytes"]
    while dimensions and dimensions[0][0] == length:
        _, size = dimensions.pop(0)
        length *= size
    for indices in itertools.product(*(range(size) for _, size in dimensions)):
        start = view["address"] + sum(
            i * stride for i, (stride, _) in zip(indices, dimensions)
        )
        yield start, start + length


def _union_size(ranges):
    total, end = 0, -1
    for start, stop in sorted(ranges):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def _segment_regions(views, start, size, block_stride):
    all_ranges, numeric = [], {}
    stop = start + size
    for view in views:
        if view["stride_bytes"][0] != block_stride:
            continue
        for lo, hi in _view_ranges(view):
            lo, hi = max(start, lo), min(stop, hi)
            if hi <= lo:
                continue
            all_ranges.append((lo - start, hi - start))
            if view["numeric"]:
                width = view["element_bytes"]
                if (lo - view["address"]) % width or (hi - lo) % width:
                    raise ValueError(f"Unaligned typed KV region for {view['name']}")
                numeric.setdefault(
                    (lo, hi, view["dtype"]),
                    {
                        "offset_bytes": lo - start,
                        "size_bytes": hi - lo,
                        "dtype": view["dtype"],
                        "name": view["name"],
                        "component": view["component"],
                    },
                )
    regions = [numeric[key] for key in sorted(numeric)]
    end = 0
    for region in regions:
        if region["offset_bytes"] < end:
            raise ValueError("Overlapping typed KV aliases need an explicit decoder")
        end = region["offset_bytes"] + region["size_bytes"]
    known = sum(region["size_bytes"] for region in regions)
    described = _union_size(all_ranges)
    if size and not described:
        raise ValueError("No registered KV view covers the Store segment")
    return regions, described - known, size - described


def build_fawa_metadata(connector, label, store_config):
    """Use the connector's exact segment sizes and ordering for one record."""
    group_ids = connector.fa_group_ids if label == "FA" else connector.window_group_ids
    groups, segments, offset = [], [], 0
    for group_id in group_ids:
        layout = connector.group_layouts.get(group_id)
        meta = connector.group_metas[group_id]
        if layout is None or not meta.tail_tokens:
            continue
        group = connector._kv_cache_config.kv_cache_groups[group_id]
        views = _views(connector.kv_caches, group)
        specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        groups.append(
            {
                "group_id": group_id,
                "token_block_size": meta.token_block_size,
                "tail_tokens": meta.tail_tokens,
                "tail_blocks": meta.tail_blocks,
                "specs": {
                    name: _spec_metadata(specs[name] if specs else group.kv_cache_spec)
                    for name in group.layer_names
                },
                "registered_views": views,
            }
        )
        sizes = layout.segment_tensor_size_list(
            meta.tail_tokens // meta.tail_blocks, meta.token_block_size
        )
        token_offset = (
            meta.token_block_size - meta.tail_tokens
            if label == "WA" and meta.tail_blocks == 1
            else 0
        )
        for tail_block in range(meta.tail_blocks):
            for column, size in enumerate(sizes):
                source = int(layout.base_ptrs[column]) + (
                    token_offset
                    * int(layout.tensor_block_sizes[column])
                    // meta.token_block_size
                ) * int(layout.tensor_token_strides[column])
                size = int(size)
                regions, opaque, padding = _segment_regions(
                    views, source, size, int(layout.block_strides[column])
                )
                segments.append(
                    {
                        "group_id": group_id,
                        "tail_block": tail_block,
                        "transfer_view": column,
                        "name": layout.view_meta[column]["name"],
                        "file_offset_bytes": offset,
                        "size_bytes": size,
                        "source_address_block_zero": source,
                        "source_block_stride_bytes": int(layout.block_strides[column]),
                        "regions": regions,
                        "opaque_bytes": opaque,
                        "padding_bytes": padding,
                    }
                )
                offset += size
    if [segment["size_bytes"] for segment in segments] != list(
        store_config["tensor_size_list"]
    ):
        raise ValueError("KV metadata segments differ from Store tensor_size_list")
    config = connector._vllm_config
    packages = {}
    for package in ("uc-manager", "vllm", "vllm-ascend", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    parallel = config.parallel_config
    pipeline = str(store_config.get("store_pipeline", ""))
    scalar_size = int(store_config.get("tensor_size", 0))
    scalar_overrides_layout = scalar_size > 0 and (
        list(store_config["tensor_size_list"])
        != [scalar_size] * (int(store_config["shard_size"]) // scalar_size)
    )
    return {
        "schema_version": 1,
        "connector": type(connector).__name__,
        "model": config.model_config.model,
        "model_dtype": str(config.model_config.dtype),
        "cache_dtype": config.cache_config.cache_dtype,
        "indexer_kv_dtype": getattr(
            getattr(config, "attention_config", None), "indexer_kv_dtype", None
        ),
        "byte_order": sys.byteorder,
        "packages": packages,
        "runtime": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "engine_id": str(connector.engine_id),
            "rank": parallel.rank,
            "dp_rank": parallel.data_parallel_rank,
            "tp_size": parallel.tensor_parallel_size,
            "pcp_size": getattr(parallel, "prefill_context_parallel_size", 1),
            "dcp_size": getattr(parallel, "decode_context_parallel_size", 1),
        },
        "hash_block_tokens": connector.hash_block_size,
        "excluded_group_ids": sorted(connector.transient_group_ids),
        "store": {
            "label": label,
            "pipeline": pipeline,
            "storage_backends": store_config.get("storage_backends", []),
            "raw_posix_supported": pipeline in ("Cache|Posix", "YuanRong|Posix")
            and not scalar_overrides_layout,
            "tensor_size_overrides_layout": scalar_overrides_layout,
            "record_size_bytes": int(connector.file_size[label]),
            "payload_size_bytes": offset,
            "trailing_padding_bytes": int(connector.file_size[label]) - offset,
        },
        "groups": groups,
        "segments": segments,
    }


def write_fawa_metadata(connector, label, store_config, directory):
    """Atomically publish one worker's store layout after Store initialization."""
    metadata = build_fawa_metadata(connector, label, store_config)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"kv-cache-{socket.gethostname()}-{os.getpid()}-{label}.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            json.dump(metadata, output, indent=2)
            output.write("\n")
            output.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def inspect_file(metadata, path):
    """Report NaN/Inf in typed regions and coverage of one immutable record."""
    if metadata["schema_version"] != 1:
        raise ValueError("Unsupported KV metadata schema")
    if not metadata["store"]["raw_posix_supported"]:
        raise ValueError(
            "This metadata does not describe an uncompressed raw Posix record"
        )
    path = Path(path)
    if path.name.endswith(".tmp"):
        raise ValueError("Inspect a completed Posix record after Dump finishes")
    result = {
        "file": str(path),
        "store": metadata["store"]["label"],
        "nan_count": 0,
        "inf_count": 0,
        "checked_float_bytes": 0,
        "integer_bytes": 0,
        "opaque_bytes": 0,
        "padding_bytes": 0,
        "issues": [],
    }
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        expected = metadata["store"]["record_size_bytes"]
        if before.st_size != expected:
            return {
                **result,
                "status": "invalid_size",
                "expected_bytes": expected,
                "actual_bytes": before.st_size,
            }
        offset = 0
        prefix = "<" if metadata["byte_order"] == "little" else ">"
        for segment in metadata["segments"]:
            size = segment["size_bytes"]
            if segment["file_offset_bytes"] != offset or size < 0:
                raise ValueError("Invalid KV metadata segment offsets")
            region_end = 0
            for region in segment["regions"]:
                start, length = region["offset_bytes"], region["size_bytes"]
                if start < region_end or length <= 0 or start + length > size:
                    raise ValueError("Invalid KV metadata typed region")
                region_end = start + length
                dtype = region["dtype"]
                if dtype in INTEGER_DTYPES:
                    result["integer_bytes"] += length
                    continue
                code, mask, limit, infinity = FLOAT_FORMATS[dtype]
                fmt = prefix + code
                width = struct.calcsize(fmt)
                if length % width:
                    raise ValueError("KV region size is not aligned to dtype")
                nan_count, inf_count, first_nan, first_inf = 0, 0, None, None
                position = offset + start
                source.seek(position)
                for chunk_start in range(position, position + length, 1 << 20):
                    count = min(1 << 20, position + length - chunk_start)
                    data = source.read(count)
                    if len(data) != count:
                        raise ValueError("Short read while inspecting KV file")
                    for index, (bits,) in enumerate(struct.iter_unpack(fmt, data)):
                        value = bits & mask
                        is_nan = value == 0x80 if limit is None else value > limit
                        if is_nan:
                            nan_count += 1
                            if first_nan is None:
                                first_nan = chunk_start + index * width
                        elif infinity is not None and value == infinity:
                            inf_count += 1
                            if first_inf is None:
                                first_inf = chunk_start + index * width
                result["checked_float_bytes"] += length
                result["nan_count"] += nan_count
                result["inf_count"] += inf_count
                if nan_count or inf_count:
                    result["issues"].append(
                        {
                            "group_id": segment["group_id"],
                            "tail_block": segment["tail_block"],
                            "name": region["name"],
                            "component": region["component"],
                            "dtype": dtype,
                            "nan_count": nan_count,
                            "inf_count": inf_count,
                            "first_nan_byte": first_nan,
                            "first_inf_byte": first_inf,
                        }
                    )
            known = sum(region["size_bytes"] for region in segment["regions"])
            if (
                min(segment["opaque_bytes"], segment["padding_bytes"]) < 0
                or known + segment["opaque_bytes"] + segment["padding_bytes"] != size
            ):
                raise ValueError("Invalid KV metadata byte coverage")
            result["opaque_bytes"] += segment["opaque_bytes"]
            result["padding_bytes"] += segment["padding_bytes"]
            offset += size
        tail = metadata["store"]["trailing_padding_bytes"]
        if (
            tail < 0
            or offset != metadata["store"]["payload_size_bytes"]
            or offset + tail != expected
        ):
            raise ValueError("Invalid KV metadata record size")
        result["padding_bytes"] += tail
        after = os.fstat(source.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("KV file changed during inspection")
    result["status"] = "no_nan_in_typed_regions"
    if result["opaque_bytes"]:
        result["status"] = "incomplete"
    if result["inf_count"]:
        result["status"] = "inf_found"
    if result["nan_count"]:
        result["status"] = "nan_found"
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    with args.metadata.open(encoding="utf-8") as source:
        metadata = json.load(source)
    exit_code = 0
    for path in args.files:
        try:
            result = inspect_file(metadata, path)
        except (OSError, ValueError, KeyError) as error:
            result = {"file": str(path), "status": "error", "error": str(error)}
        print(json.dumps(result, ensure_ascii=False))
        if result["status"] != "no_nan_in_typed_regions":
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
