# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Thread-safe in-memory mock of the pynds_kv Python API.

This module is for Python integration and scheduling tests only. It neither
imports the native extension nor emulates NDS/RDMA transport behavior.
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass

NDS_SUCCESS = 0
NDS_FAILED = 1
_MAX_BLOCK_ID = 1 << 80
_MAX_NSID = 1 << 32


@dataclass(frozen=True)
class Call:
    """One dispatched mock API call, exposed for integration assertions."""

    operation: str
    thread_id: int


_lock = threading.RLock()
_initialized = False
_store = {}
_calls = []


def reset():
    """Clear data, initialization state, and call history between tests."""

    global _initialized
    with _lock:
        _initialized = False
        _store.clear()
        _calls.clear()


def get_calls():
    """Return a snapshot of API calls, including their dispatch thread IDs."""

    with _lock:
        return list(_calls)


def _record(operation):
    _calls.append(Call(operation=operation, thread_id=threading.get_ident()))


def _validate_block_id(block_id):
    if not isinstance(block_id, int) or not 0 <= block_id < _MAX_BLOCK_ID:
        raise ValueError("block_id must be a non-negative int < 2^80")
    return block_id


def _validate_block_ids(block_ids):
    return [_validate_block_id(block_id) for block_id in block_ids]


def _validate_nsid(nsid):
    if not isinstance(nsid, int) or not 0 <= nsid < _MAX_NSID:
        raise ValueError("nsid must be an int in [0, 2^32)")
    return nsid


def _validate_offset_length(offset, length):
    if not isinstance(offset, int) or not isinstance(length, int):
        raise TypeError("offset and length must be ints")
    if offset < 0 or length < 0:
        raise ValueError("offset and length must be non-negative")


@contextmanager
def _byte_view(buffer):
    original = memoryview(buffer)
    view = original
    try:
        if not original.contiguous:
            raise TypeError("buffer must be contiguous")
        if original.format != "B" or original.ndim != 1:
            view = original.cast("B")
        yield view
    finally:
        if view is not original:
            view.release()
        original.release()


def _validate_buffer_range(buffer, offset, length, write=False):
    _validate_offset_length(offset, length)
    with _byte_view(buffer) as view:
        if offset > view.nbytes or length > view.nbytes - offset:
            raise ValueError("Buffer is too small for offset + length")
        if write and view.readonly:
            raise TypeError("destination buffer must be writable")


def _read_payload(buffer, offset, length):
    _validate_buffer_range(buffer, offset, length)
    with _byte_view(buffer) as view:
        return view[offset : offset + length].tobytes()


def _write_payload(buffer, offset, length, payload):
    _validate_buffer_range(buffer, offset, length, write=True)
    with _byte_view(buffer) as view:
        view[offset : offset + length] = payload[:length]


def _validate_registration(buffer, length):
    _validate_buffer_range(buffer, 0, length)
    return length != 0


def init(buffer, length, nds_config_path=None):
    """Mock registration of one buffer; the configuration path is ignored."""

    del nds_config_path
    valid_length = _validate_registration(buffer, length)
    global _initialized
    with _lock:
        _record("init")
        if not valid_length:
            return NDS_FAILED
        _initialized = True
    return NDS_SUCCESS


def init_multi(buffers, lengths, nds_config_path=None):
    """Mock registration of multiple buffers; list order represents mr_id."""

    del nds_config_path
    buffers = list(buffers)
    lengths = list(lengths)
    if not buffers:
        raise ValueError("buffers must not be empty")
    if len(buffers) != len(lengths):
        raise ValueError("buffers and lengths must have same length")
    valid = True
    for buffer, length in zip(buffers, lengths):
        valid = _validate_registration(buffer, length) and valid

    global _initialized
    with _lock:
        _record("init_multi")
        if not valid:
            return NDS_FAILED
        _initialized = True
    return NDS_SUCCESS


def is_exists(block_ids, nsid=0):
    """Return the consecutive prefix hit count within one namespace."""

    block_ids = _validate_block_ids(block_ids)
    _validate_nsid(nsid)
    with _lock:
        _record("is_exists")
        if not _initialized:
            return -1
        hits = 0
        for block_id in block_ids:
            if (nsid, block_id) not in _store:
                break
            hits += 1
        return hits


def put(block_id, buffer, offset, length, nsid):
    """Copy one buffer range into the in-memory namespace store."""

    block_id = _validate_block_id(block_id)
    _validate_nsid(nsid)
    payload = _read_payload(buffer, offset, length)
    with _lock:
        _record("put")
        if not _initialized:
            return NDS_FAILED
        _store[(nsid, block_id)] = payload
    return NDS_SUCCESS


def get(block_id, buffer, offset, length, nsid):
    """Copy one stored value into a writable buffer range."""

    block_id = _validate_block_id(block_id)
    _validate_nsid(nsid)
    _validate_buffer_range(buffer, offset, length, write=True)
    with _lock:
        _record("get")
        if not _initialized:
            return NDS_FAILED
        payload = _store.get((nsid, block_id))
        if payload is None or len(payload) < length:
            return NDS_FAILED
        _write_payload(buffer, offset, length, payload)
    return NDS_SUCCESS


def _validate_batch(block_ids, buffers, offsets, lengths, nsids, write=False):
    block_ids = _validate_block_ids(block_ids)
    buffers = list(buffers)
    offsets = list(offsets)
    lengths = list(lengths)
    nsids = list(nsids)
    count = len(block_ids)
    if not all(len(items) == count for items in (buffers, offsets, lengths, nsids)):
        raise ValueError(
            "block_ids, buffers, offsets, lengths, nsids must have same length"
        )
    for buffer, offset, length, nsid in zip(buffers, offsets, lengths, nsids):
        _validate_nsid(nsid)
        _validate_buffer_range(buffer, offset, length, write=write)
    return block_ids, buffers, offsets, lengths, nsids


def batchPut(block_ids, buffers, offsets, lengths, nsids):
    """Dispatch a batch of writes after validating every argument list."""

    block_ids, buffers, offsets, lengths, nsids = _validate_batch(
        block_ids, buffers, offsets, lengths, nsids
    )
    payloads = [
        _read_payload(buffer, offset, length)
        for buffer, offset, length in zip(buffers, offsets, lengths)
    ]
    with _lock:
        _record("batchPut")
        if not _initialized:
            return NDS_FAILED
        for block_id, nsid, payload in zip(block_ids, nsids, payloads):
            _store[(nsid, block_id)] = payload
    return NDS_SUCCESS


def batchGet(block_ids, buffers, offsets, lengths, nsids):
    """Dispatch a batch of reads, failing if any requested key is absent."""

    block_ids, buffers, offsets, lengths, nsids = _validate_batch(
        block_ids, buffers, offsets, lengths, nsids, write=True
    )
    with _lock:
        _record("batchGet")
        if not _initialized:
            return NDS_FAILED
        payloads = []
        for block_id, length, nsid in zip(block_ids, lengths, nsids):
            payload = _store.get((nsid, block_id))
            if payload is None or len(payload) < length:
                return NDS_FAILED
            payloads.append(payload)
        for buffer, offset, length, payload in zip(buffers, offsets, lengths, payloads):
            _write_payload(buffer, offset, length, payload)
    return NDS_SUCCESS


__all__ = [
    "Call",
    "NDS_FAILED",
    "NDS_SUCCESS",
    "batchGet",
    "batchPut",
    "get",
    "get_calls",
    "init",
    "init_multi",
    "is_exists",
    "put",
    "reset",
]
