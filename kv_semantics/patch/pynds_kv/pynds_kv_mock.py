# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Small in-memory pynds_kv test double; no NDS hardware is required."""

import ctypes

_PAGE_SIZE = 4096
_data = {}
_initialized = False


def _is_valid_registration(buffer, length):
    if length == 0 or length > memoryview(buffer).nbytes:
        return False
    address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    return address % _PAGE_SIZE == 0


def init(buffer, length, nds_config_path=None):
    return init_multi([buffer], [length], nds_config_path)


def init_multi(buffers, lengths, nds_config_path=None):
    if not buffers or len(buffers) != len(lengths):
        raise ValueError("buffers and lengths must be non-empty and have equal length")
    if not all(
        _is_valid_registration(buffer, length)
        for buffer, length in zip(buffers, lengths)
    ):
        return 1
    global _initialized
    _initialized = True
    return 0


def is_exists(block_ids, nsid=0):
    if not _initialized:
        return -1
    hits = 0
    for block_id in block_ids:
        if (nsid, block_id) not in _data:
            break
        hits += 1
    return hits


def put(block_id, buffer, offset, length, nsid):
    if not _initialized:
        return 1
    _data[(nsid, block_id)] = memoryview(buffer)[offset : offset + length].tobytes()
    return 0


def get(block_id, buffer, offset, length, nsid):
    payload = _data.get((nsid, block_id))
    if not _initialized or payload is None or len(payload) < length:
        return 1
    memoryview(buffer)[offset : offset + length] = payload[:length]
    return 0


def batchPut(block_ids, buffers, offsets, lengths, nsids):
    for args in zip(block_ids, buffers, offsets, lengths, nsids):
        result = put(*args)
        if result != 0:
            return result
    return 0


def batchGet(block_ids, buffers, offsets, lengths, nsids):
    for args in zip(block_ids, buffers, offsets, lengths, nsids):
        result = get(*args)
        if result != 0:
            return result
    return 0
