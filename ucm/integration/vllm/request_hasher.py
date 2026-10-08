import ast
import hashlib
import inspect
import pickle
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.request import Request

try:
    from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys
except ImportError:  # pragma: no cover - depends on the installed vLLM version
    generate_block_hash_extra_keys = None


class RequestHashError(RuntimeError):
    """Raised when UCM cannot safely hash all KV-affecting request semantics."""


def _request_has_extra_hash_semantics(request: "Request") -> bool:
    return bool(
        getattr(request, "mm_features", None)
        or getattr(request, "lora_request", None) is not None
        or getattr(request, "cache_salt", None)
        or getattr(request, "prompt_embeds", None) is not None
    )


def _validate_request_hash_semantics(request: "Request") -> None:
    if generate_block_hash_extra_keys is None and _request_has_extra_hash_semantics(
        request
    ):
        raise RequestHashError(
            "The installed vLLM does not expose "
            "generate_block_hash_extra_keys(), but this request contains "
            "KV-affecting semantics beyond token IDs."
        )


def _generate_extra_keys(
    request: "Request",
    start_token_idx: int,
    end_token_idx: int,
    start_mm_idx: int,
) -> tuple[tuple[Any, ...] | None, int]:
    if generate_block_hash_extra_keys is None:
        return None, start_mm_idx

    try:
        return generate_block_hash_extra_keys(
            request,
            start_token_idx,
            end_token_idx,
            start_mm_idx,
        )
    except Exception as exc:
        raise RequestHashError(
            "Failed to generate vLLM block hash extra keys for UCM: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _collect_layout_classes() -> list[type]:
    """KVCacheLayout subclasses (recursive) plus the standalone HMA layout.
    The explicit imports pin the class set so scheduler and worker compute
    an identical digest.
    """
    import ucm.integration.vllm.hla_connector
    import ucm.integration.vllm.hma_connector as hma_module
    import ucm.integration.vllm.ucm_connector as ucm_module

    classes: set[type] = set()

    def walk(cls: type) -> None:
        classes.add(cls)
        for subclass in cls.__subclasses__():
            walk(subclass)

    walk(ucm_module.KVCacheLayout)
    classes.add(hma_module.KVCacheGroupLayout)
    return sorted(classes, key=lambda cls: f"{cls.__module__}.{cls.__qualname__}")


def _layout_source_digest() -> str:
    """Digest of the normalized source of all layout classes; any
    behavior-relevant layout change re-keys every block hash. Falls back
    to the installed package version when sources are unavailable."""
    try:
        digest = hashlib.sha256()
        for cls in _collect_layout_classes():
            source = inspect.getsource(cls)
            tree = ast.parse(textwrap.dedent(source))
            digest.update(f"{cls.__module__}.{cls.__qualname__}:".encode("utf-8"))
            digest.update(ast.dump(tree, include_attributes=False).encode("utf-8"))
        return digest.hexdigest()[:16]
    except OSError:
        # Sourceless deployment: fall back to per-version isolation.
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("uc-manager")
        except PackageNotFoundError:
            return "unknown"


class RequestHasher:
    """Generate stable, namespaced UCM request and block identifiers."""

    def __init__(self, vllm_config, rank_id, kv_cache_config=None):
        speculative_config = getattr(vllm_config, "speculative_config", None)
        spec_info = ""
        if speculative_config is not None:
            spec_method = getattr(speculative_config, "method", "") or ""
            spec_tokens = getattr(speculative_config, "num_speculative_tokens", 0)
            spec_info = f":{spec_method}:{spec_tokens}"
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        sparse_sfa_c8 = bool(additional_config.get("enable_sparse_sfa_c8", False))
        sparse_li_c8 = bool(additional_config.get("enable_sparse_li_c8", False))
        sparse_c8_info = f":sfa_c8={int(sparse_sfa_c8)}:li_c8={int(sparse_li_c8)}"
        hf_config = getattr(vllm_config.model_config, "hf_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None)
        is_v41_layout = bool(
            getattr(hf_config, "kv_source_layer_ids", None)
            or getattr(hf_config, "index_source_layer_ids", None)
            or (
                compress_ratios is not None
                and {int(ratio) for ratio in compress_ratios}.issubset({0, 1, 2})
                and bool({int(ratio) for ratio in compress_ratios} & {1, 2})
            )
        )
        layout_info = ""
        if is_v41_layout:
            cache_config = getattr(vllm_config, "cache_config", None)
            if cache_config is None:
                raise ValueError("DeepSeek V4.1 cache namespace needs cache_config.")
            ratios = ",".join(str(int(ratio)) for ratio in compress_ratios or ())
            layout_info = (
                ":kv_layout=v41_packed_v3"
                f":cache_dtype={cache_config.cache_dtype}"
                f":block_size={cache_config.block_size}"
                f":compress_ratios={ratios}"
            )
            if kv_cache_config is not None:
                tensors = getattr(kv_cache_config, "kv_cache_tensors", ())
                has_placement = bool(tensors) and hasattr(tensors[0], "layer_stride")
                groups = []
                for group in kv_cache_config.kv_cache_groups:
                    spec = group.kv_cache_spec
                    nested = getattr(spec, "kv_cache_specs", None)
                    layers = []
                    for name in sorted(group.layer_names):
                        if has_placement:
                            layers.append(name)
                            continue
                        inner = nested[name] if nested else spec
                        layers.append(
                            (
                                name,
                                getattr(inner, "tokens_per_state", 1),
                                getattr(inner, "state_content_size_bytes", None),
                                str(getattr(inner, "dtype", None)),
                                str(getattr(inner, "kv_quant_mode", None)),
                            )
                        )
                    groups.append(
                        (
                            spec.block_size,
                            getattr(spec, "prefix_cacheable", True),
                            getattr(spec, "prefix_replay_tokens", 0),
                            getattr(group, "enable_kv_transfer", True),
                            tuple(layers),
                        )
                    )
                layout_info += f":transfer_layout=v4:{tuple(groups)!r}"
                if has_placement:
                    placement = tuple(
                        (tuple(t.layers), t.layer_stride, t.block_stride, t.offset)
                        for t in tensors
                    )
                    layout_info += f":placement={placement!r}"
        model_name = vllm_config.model_config.model.rstrip("/").split("/")[-1]
        dcp_size = getattr(
            vllm_config.parallel_config, "decode_context_parallel_size", 1
        )
        dcp_info = f":dcp={dcp_size}"
        layout_source_info = f":layout={_layout_source_digest()}"
        meta = (
            f"{model_name}:"
            f"{vllm_config.parallel_config.tensor_parallel_size}:"
            f"{vllm_config.model_config.dtype}:{rank_id}{spec_info}"
            f"{sparse_c8_info}{dcp_info}{layout_source_info}{layout_info}"
        )
        self.meta_bytes = meta.encode("utf-8")
        self.seed = self("UCM_HASH_SEED")

    def __call__(self, input_data) -> bytes:
        if isinstance(input_data, bytes):
            input_bytes = input_data
        else:
            input_bytes = pickle.dumps(input_data, protocol=pickle.HIGHEST_PROTOCOL)

        h = hashlib.md5(self.meta_bytes + input_bytes)
        return h.digest()

    def make_request_block_hasher(
        self,
        block_size: int,
        initial_hash: bytes | None = None,
    ) -> Callable[["Request"], list[bytes]]:
        """Bind a reusable request hasher to one block size and chain root."""
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}.")

        root = self.seed if initial_hash is None else initial_hash

        def hash_request(request: "Request") -> list[bytes]:
            _validate_request_hash_semantics(request)
            token_ids = request.all_token_ids
            parent = root
            curr_mm_idx = 0
            block_hashes: list[bytes] = []

            for start in range(0, len(token_ids), block_size):
                end = start + block_size
                if end > len(token_ids):
                    break

                extra_keys, curr_mm_idx = _generate_extra_keys(
                    request,
                    start,
                    end,
                    curr_mm_idx,
                )
                parent = self(
                    (
                        parent,
                        tuple(token_ids[start:end]),
                        extra_keys,
                    )
                )
                block_hashes.append(parent)

            return block_hashes

        return hash_request


__all__ = ["RequestHashError", "RequestHasher"]
