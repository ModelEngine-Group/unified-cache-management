import hashlib
import pickle
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


class RequestHasher:
    """Generate stable, namespaced UCM request and block identifiers."""

    def __init__(self, vllm_config, rank_id, *, namespace=None):
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
        model_name = vllm_config.model_config.model.rstrip("/").split("/")[-1]
        meta = (
            f"{model_name}:"
            f"{vllm_config.parallel_config.tensor_parallel_size}:"
            f"{vllm_config.model_config.dtype}:{rank_id}{spec_info}{sparse_c8_info}"
        )
        self.meta_bytes = meta.encode("utf-8")
        if namespace is not None:
            self.meta_bytes += pickle.dumps(namespace, protocol=4)
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


def encode_block_key(digest: bytes, group_id: int = 0) -> bytes:
    """Create a rank-0 key: 112 digest bits, reserved=0, group:4, rank:4."""
    if len(digest) != 16:
        raise ValueError("A block digest/key must contain exactly 16 bytes.")
    if not isinstance(group_id, int) or not 0 <= group_id < 16:
        raise ValueError("group_id must be an integer in [0, 15].")
    return digest[:14] + bytes((0, group_id << 4))


def set_block_key_rank(key: bytes, rank_id: int) -> bytes:
    """Replace only the rank nibble of an encoded key; preserve all other bits."""
    if len(key) != 16:
        raise ValueError("A block key must contain exactly 16 bytes.")
    if not isinstance(rank_id, int) or not 0 <= rank_id < 16:
        raise ValueError("rank_id must be an integer in [0, 15].")
    return key[:15] + bytes(((key[15] & 0xF0) | rank_id,))


def kv_layout_namespace(vllm_config, kv_cache_config=None) -> tuple:
    """Describe serialization, without addresses or allocated block counts.

    Use resolved per-layer specs when available: model dtype alone does not
    distinguish FP8 KV from BF16 KV, or FP16 scales from FP32 scales.
    """
    fields = (
        "block_size",
        "storage_block_size",
        "num_kv_heads",
        "head_size",
        "dtype",
        "scale_dim",
        "scale_dtype",
        "compress_ratio",
        "sliding_window",
        "shapes",
        "dtypes",
        "mamba_cache_mode",
        "page_size_padded",
        "cache_sparse_sfa_c8",
    )
    groups = []
    for group in getattr(kv_cache_config, "kv_cache_groups", ()):
        spec = group.kv_cache_spec
        nested = getattr(spec, "kv_cache_specs", None)
        members = []
        for name in sorted(group.layer_names):
            member = nested[name] if nested else spec
            members.append(
                (
                    name,
                    type(member).__name__,
                    tuple(
                        (field, str(getattr(member, field)))
                        for field in fields
                        if hasattr(member, field)
                    ),
                )
            )
        groups.append(tuple(members))
    if groups:
        return tuple(groups)
    cache_config = getattr(vllm_config, "cache_config", None)
    return (
        "config-fallback",
        str(getattr(cache_config, "cache_dtype", "auto")),
        getattr(cache_config, "block_size", None),
    )


__all__ = [
    "RequestHashError",
    "RequestHasher",
    "encode_block_key",
    "set_block_key_rank",
]
