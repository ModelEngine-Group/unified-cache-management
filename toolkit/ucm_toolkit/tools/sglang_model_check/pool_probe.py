"""Configuration-derived physical pool plans; never substitute generic side pools."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any


class PoolProbeUnavailable(ValueError):
    """The installed SGLang cannot describe this model's physical pools offline."""


@dataclass(frozen=True)
class PoolSpec:
    name: str
    kind: str
    layers: tuple[int, ...]
    window_pages: int = 0
    head_num: int = 0
    head_dim: int = 0
    v_head_dim: int = 0
    kv_cache_dim: int = 0
    temporal_shape: tuple[int, ...] = ()
    conv_shapes: tuple[tuple[int, ...], ...] = ()
    temporal_dtype: Any = None
    conv_dtype: Any = None
    storage_page_size: int = 0
    page_bytes: int = 0
    source: str = "config"

    def report(self) -> dict[str, Any]:
        result = {
            "kind": self.kind,
            "layer_ids": list(self.layers),
            "window_pages": self.window_pages,
            "source": self.source,
        }
        if self.kind in {"native", "logical"}:
            result.update(page_size=self.storage_page_size, page_bytes=self.page_bytes)
            return result
        if self.kind == "mla":
            result.update(
                kv_cache_dim=self.kv_cache_dim, page_size=self.storage_page_size
            )
        elif self.kind == "mamba":
            result.update(
                temporal_shape=list(self.temporal_shape),
                conv_shapes=[list(shape) for shape in self.conv_shapes],
                temporal_dtype=str(self.temporal_dtype),
                conv_dtype=str(self.conv_dtype),
                page_size=1,
            )
        else:
            result.update(
                head_num=self.head_num,
                head_dim=self.head_dim,
                v_head_dim=self.v_head_dim,
            )
        return result

    def bytes_per_slot(self, model_dtype: Any, page_size: int) -> int:
        if self.kind in {"native", "logical"}:
            return self.page_bytes
        if self.kind == "mla":
            return (
                len(self.layers) * page_size * self.kv_cache_dim * model_dtype.itemsize
            )
        if self.kind == "mamba":
            per_layer = prod(self.temporal_shape) * self.temporal_dtype.itemsize
            per_layer += (
                sum(prod(s) for s in self.conv_shapes) * self.conv_dtype.itemsize
            )
            return len(self.layers) * per_layer
        return (
            len(self.layers)
            * page_size
            * self.head_num
            * (self.head_dim + self.v_head_dim)
            * model_dtype.itemsize
        )


def _layers(value: Any, label: str) -> tuple[int, ...]:
    if value is None:
        raise PoolProbeUnavailable(f"SGLang did not expose {label}")
    result = tuple(int(item) for item in value)
    if len(set(result)) != len(result):
        raise PoolProbeUnavailable(f"duplicate layer IDs in {label}")
    return result


def build_pool_specs(
    model_config: Any,
    pool_names: tuple[str, ...],
    page_size: int,
    platform: str,
    *,
    linear_config: Any = None,
    external_attention: bool = False,
    tp_size: int = 1,
    layer_range: tuple[int, int] | None = None,
) -> list[PoolSpec]:
    """Reuse SGLang's layer lists, head geometry and mamba2_cache_params.

    Scope: physical MHA/MLA anchor and standard persistent linear state; MHA SWA,
    one rank, no speculative/replay side state, unquantized model-dtype KV.
    ``linear_config`` is injectable for dependency-free plan regression tests.
    """
    unsupported = set(pool_names) - {"kv", "swa", "mamba"}
    if unsupported or "kv" not in pool_names:
        raise PoolProbeUnavailable(
            f"physical descriptors are not implemented for pools {sorted(unsupported or set(pool_names))}"
        )
    arch = getattr(
        model_config.attention_arch, "name", str(model_config.attention_arch)
    )
    if arch.upper() not in {"MHA", "MLA"}:
        raise PoolProbeUnavailable(f"unsupported hybrid attention architecture: {arch}")
    if arch.upper() == "MLA" and "swa" in pool_names:
        raise PoolProbeUnavailable(
            "MLA SWA requires specialized physical pool descriptors"
        )
    if getattr(model_config, "is_hybrid_swa_compress", False):
        raise PoolProbeUnavailable(
            "compressed/heterogeneous SWA requires runtime buffer descriptors"
        )

    total = int(model_config.num_hidden_layers)
    linear_layers: tuple[int, ...] = ()
    params = None
    if "mamba" in pool_names:
        if linear_config is None:
            from sglang.srt.configs.hybrid_arch import mambaish_config

            linear_config = mambaish_config(model_config)
        if linear_config is None:
            raise PoolProbeUnavailable(
                "SGLang mambaish_config returned no state config"
            )
        params = linear_config.mamba2_cache_params
        linear_layers = _layers(params.layers, "mamba2_cache_params.layers")
        full = _layers(linear_config.full_attention_layer_ids, "full attention layers")
    else:
        full = _layers(model_config.full_attention_layer_ids, "full attention layers")

    swa = (
        _layers(model_config.swa_attention_layer_ids, "SWA layers")
        if "swa" in pool_names
        else ()
    )
    # Refuse overlapping or incomplete partitions instead of copying all model
    # layers into every pool. This is important for three-way hybrid layouts.
    combined = (*full, *swa, *linear_layers)
    if len(combined) != len(set(combined)) or set(combined) != set(range(total)):
        raise PoolProbeUnavailable(
            "SGLang layer lists are not a complete disjoint KV/SWA/state partition"
        )
    if layer_range is not None:
        start, end = layer_range
        if not 0 <= start < end <= total:
            raise PoolProbeUnavailable("invalid PP layer range")
        full = tuple(i for i in full if start <= i < end)
        swa = tuple(i for i in swa if start <= i < end)
        linear_layers = tuple(i for i in linear_layers if start <= i < end)
    specs = []
    for name, layers in (("kv", full), ("swa", swa)):
        if name not in pool_names:
            continue
        if not layers:
            if name == "kv":
                specs.append(PoolSpec("kv", "logical", (), storage_page_size=page_size))
                continue
            if layer_range is None:
                raise PoolProbeUnavailable(f"{name} has no physical layers")
            continue  # This PP stage may contain no layers of this side pool.
        if external_attention:
            if name != "kv" or arch.upper() != "MLA":
                raise PoolProbeUnavailable(
                    "external attention descriptors require a single MLA anchor"
                )
            # Geometry is supplied by the real DSA child constructor, not by
            # MHA head arithmetic. Keep only its verified global layer IDs.
            specs.append(PoolSpec(name, "native", layers, storage_page_size=page_size))
            continue
        if arch.upper() == "MLA":
            width = int(model_config.kv_lora_rank) + int(model_config.qk_rope_head_dim)
            if width <= 0:
                raise PoolProbeUnavailable("invalid MLA latent cache dimension")
            specs.append(
                PoolSpec(
                    name, "mla", layers, kv_cache_dim=width, storage_page_size=page_size
                )
            )
            continue
        is_swa = name == "swa"
        head_getter = (
            model_config.get_swa_num_kv_heads
            if is_swa
            else model_config.get_num_kv_heads
        )
        head_dim = (
            int(getattr(model_config, "swa_head_dim", None) or model_config.head_dim)
            if is_swa
            else int(model_config.head_dim)
        )
        v_dim = (
            int(getattr(model_config, "swa_v_head_dim", None) or head_dim)
            if is_swa
            else int(getattr(model_config, "v_head_dim", None) or head_dim)
        )
        window_pages = 0
        if is_swa:
            window = model_config.sliding_window_size
            if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
                raise PoolProbeUnavailable(f"unsupported SWA window {window!r}")
            window_pages = (window + page_size - 1) // page_size
        specs.append(
            PoolSpec(
                name,
                "mha",
                layers,
                window_pages=window_pages,
                head_num=int(head_getter(tp_size)),
                head_dim=head_dim,
                v_head_dim=v_dim,
            )
        )

    if params is not None and linear_layers:
        temporal = tuple(int(dim) for dim in params.shape.temporal)
        conv = tuple(tuple(int(dim) for dim in shape) for shape in params.shape.conv)
        if not linear_layers or not conv or any(dim < 0 for dim in temporal):
            raise PoolProbeUnavailable("invalid/empty Mamba state description")
        if any(not shape or any(dim <= 0 for dim in shape) for shape in conv):
            raise PoolProbeUnavailable("invalid/empty Mamba convolution state")
        # Mirror the production NPU persistent conv layout (no speculative
        # extension). KDA already describes [window, channels]. CPU AMX uses
        # another layout and cannot be inferred from these config shapes alone.
        if platform == "ascend" and not getattr(params, "is_kda", False):
            if any(len(shape) != 2 for shape in conv):
                raise PoolProbeUnavailable(
                    "NPU conv layout requires two-dimensional shapes"
                )
            conv = tuple((shape[1], shape[0]) for shape in conv)
        if platform == "cpu":
            raise PoolProbeUnavailable(
                "CPU linear-state runtime layout needs AMX-specific descriptors"
            )
        specs.append(
            PoolSpec(
                "mamba",
                "mamba",
                linear_layers,
                window_pages=1,
                temporal_shape=temporal,
                conv_shapes=conv,
                temporal_dtype=params.dtype.temporal,
                conv_dtype=params.dtype.conv,
            )
        )
    if any(spec.kind == "mha" and spec.head_dim != spec.v_head_dim for spec in specs):
        raise PoolProbeUnavailable(
            "asymmetric K/V requires a specialized physical host pool"
        )
    if any(
        spec.kind == "mha" and min(spec.head_num, spec.head_dim) <= 0 for spec in specs
    ):
        raise PoolProbeUnavailable("invalid attention head geometry")
    return specs


def checkpoint_pages(pages: int) -> list[int]:
    """Sparse checkpoints for multi-page probes; always include the last page."""
    return sorted({pages, pages // 2} - {0}) if pages > 2 else [pages]


def persisted_positions(pages: int, window: int, checkpoints: list[int]) -> list[int]:
    if not window:
        return list(range(pages))
    return sorted(
        {index for end in checkpoints for index in range(max(0, end - window), end)}
    )


def expected_prefixes(
    pages: int, written: dict[str, set[int]], windows: dict[str, int]
) -> list[int]:
    """Reference oracle, independent of the connector's lookup implementation."""
    result = []
    for end in range(1, pages + 1):
        if all(
            all(
                index in positions
                for index in range(
                    max(0, end - windows[name]) if windows[name] else 0, end
                )
            )
            for name, positions in written.items()
        ):
            result.append(end)
    return result
