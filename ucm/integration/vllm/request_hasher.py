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
    """Hash using a caller-provided deployment fingerprint."""

    def __init__(self, fingerprint: bytes):
        self.meta_bytes = fingerprint
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
        """Return 14-byte digest prefixes with two zero metadata bytes.

        Keep full MD5 digests internally as chain parents; group/rank setters
        only modify the returned keys, never the content chain.
        """
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
                block_hashes.append(key_from_digest(parent))

            return block_hashes

        return hash_request


def key_from_digest(digest: bytes) -> bytes:
    """Create a key with a 112-bit digest prefix and zeroed metadata."""
    return digest[:14] + bytes(2)


def set_group_id(key: bytes, group_id: int) -> bytes:
    """Replace the group nibble; preserve the digest, reserved bits and rank."""
    if len(key) != 16:
        raise ValueError("A block key must contain exactly 16 bytes.")
    if not isinstance(group_id, int) or not 0 <= group_id < 16:
        raise ValueError("group_id must be an integer in [0, 15].")
    return key[:15] + bytes(((group_id << 4) | (key[15] & 0x0F),))


def set_rank_id(key: bytes, rank_id: int) -> bytes:
    """Replace the rank nibble; preserve the digest, reserved bits and group."""
    if len(key) != 16:
        raise ValueError("A block key must contain exactly 16 bytes.")
    if not isinstance(rank_id, int) or not 0 <= rank_id < 16:
        raise ValueError("rank_id must be an integer in [0, 15].")
    return key[:15] + bytes(((key[15] & 0xF0) | rank_id,))


__all__ = [
    "RequestHashError",
    "RequestHasher",
    "key_from_digest",
    "set_group_id",
    "set_rank_id",
]
