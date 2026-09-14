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

    def __init__(self, vllm_config, rank_id):
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
