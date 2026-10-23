"""UCM chunk id encoding.

A UCM chunk id is the key of one Store record. Only this module knows how
an id is laid out. The scheduler cuts chunk hashes into common UCM chunk ids
with make_common_ucm_chunk_ids; each rank appends its tag with
make_local_ucm_chunk_ids to get the local UCM chunk ids it stores.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import NamedTuple

import numpy as np

# Hashed into every id, so ids of different versions never collide.
# Bump it when the id or record format changes.
UCM_CHUNK_ID_VERSION = 1


class AttentionType(enum.Enum):
    """Kind of data a UCM record stores. The value is written to dump metadata."""

    FA = "FA"  # full-attention groups
    WA = "WA"  # sliding-window groups that store a window tail
    STATE = "State"  # mamba state snapshot groups


class ShardKey(NamedTuple):
    """Which part of the data a rank holds.

    head_key is the KV head shard (TP), token_key is the token shard (DCP) and
    pp_key is the pipeline stage. Ranks with the same key hold the same data.
    """

    head_key: int = 0
    token_key: int = 0
    pp_key: int = 0


# A local UCM chunk id is a 12-byte common UCM chunk id + a 4-byte big-endian
# tag. The common id is the same on every rank; the tag says which record
# and shard this rank stores. Tag bits from high to low: type 2, record
# class 4, pp 5, head 6, token 7, reserved 8. Each field below is
# (bits, shift).
_COMMON_BYTES = 12
_TAG_BYTES = 4
_TOKEN_FIELD = (7, 8)
_HEAD_FIELD = (6, _TOKEN_FIELD[1] + _TOKEN_FIELD[0])
_PP_FIELD = (5, _HEAD_FIELD[1] + _HEAD_FIELD[0])
_RECORD_CLASS_FIELD = (4, _PP_FIELD[1] + _PP_FIELD[0])
_TYPE_FIELD = (2, _RECORD_CLASS_FIELD[1] + _RECORD_CLASS_FIELD[0])
assert sum(_TYPE_FIELD) == _TAG_BYTES * 8
_TYPE_CODES = {AttentionType.FA: 0, AttentionType.WA: 1, AttentionType.STATE: 2}


def _field(value: int, field: tuple[int, int]) -> int:
    bits, shift = field
    return (value & ((1 << bits) - 1)) << shift


def validate_shard_key_encoding(record_class: int, shard_key: ShardKey) -> None:
    """Raise at startup if the topology does not fit in the id fields."""
    for field_name, field_value, (bits, _shift) in (
        ("record class", record_class, _RECORD_CLASS_FIELD),
        ("pp shard", shard_key.pp_key, _PP_FIELD),
        ("head shard", shard_key.head_key, _HEAD_FIELD),
        ("token shard", shard_key.token_key, _TOKEN_FIELD),
    ):
        if 0 <= field_value < 1 << bits:
            continue
        else:
            raise ValueError(f"UCM chunk ids encode at most {1 << bits} {field_name}s")


@lru_cache(maxsize=None)
def _tag(
    attention_type: AttentionType, record_class: int, shard_key: ShardKey
) -> bytes:
    return (
        _field(_TYPE_CODES[attention_type], _TYPE_FIELD)
        | _field(record_class, _RECORD_CLASS_FIELD)
        | _field(shard_key.pp_key, _PP_FIELD)
        | _field(shard_key.head_key, _HEAD_FIELD)
        | _field(shard_key.token_key, _TOKEN_FIELD)
    ).to_bytes(_TAG_BYTES, "big")


def make_common_ucm_chunk_ids(chunk_hashes: Iterable[bytes]) -> tuple[bytes, ...]:
    """Common UCM chunk ids: the first bytes of each chunk hash.

    They are the same on every rank. Each rank appends its own tag to get
    its local UCM chunk ids.
    """
    return tuple(chunk_hash[:_COMMON_BYTES] for chunk_hash in chunk_hashes)


def make_local_ucm_chunk_ids(
    common_ucm_chunk_ids: Sequence[bytes],
    attention_type: AttentionType,
    record_class: int = 0,
    shard_key: ShardKey = ShardKey(),
) -> tuple[bytes, ...]:
    """Local UCM chunk ids of these chunks for one record class and shard."""
    if all(len(common_id) == _COMMON_BYTES for common_id in common_ucm_chunk_ids):
        tag = _tag(attention_type, record_class, ShardKey(*shard_key))
        return tuple(
            common_ucm_chunk_id + tag for common_ucm_chunk_id in common_ucm_chunk_ids
        )
    else:
        raise ValueError(
            f"Common UCM chunk ids must be {_COMMON_BYTES} bytes; "
            "build them with make_common_ucm_chunk_ids"
        )


def writer_rows(
    common_ucm_chunk_ids: Sequence[bytes], replica_index: int, replica_count: int
) -> np.ndarray:
    """Rows this replica writes.

    Chunks are assigned to replicas by common UCM chunk id. Every replica
    computes the same assignment, so each chunk is written exactly once.
    """
    writer_indices = np.fromiter(
        (
            int.from_bytes(common_ucm_chunk_id[:8], "little") % replica_count
            for common_ucm_chunk_id in common_ucm_chunk_ids
        ),
        dtype=np.int64,
        count=len(common_ucm_chunk_ids),
    )
    return np.flatnonzero(writer_indices == replica_index)
