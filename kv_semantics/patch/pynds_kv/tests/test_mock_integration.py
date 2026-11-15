# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Run a real byte round trip through the mock, without NDS hardware."""

import mmap
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pynds_kv_mock


def main():
    with patch.dict(sys.modules, {"pynds_kv": pynds_kv_mock}):
        import pynds_kv

        unaligned = memoryview(bytearray(4097))[1:]
        assert pynds_kv.init(unaligned, 4096) != 0

        block_ids = [1001, 1002]
        block_size = 64
        nsid = 9
        storage = mmap.mmap(-1, block_size * len(block_ids))
        blocks = [
            memoryview(storage)[index * block_size : (index + 1) * block_size]
            for index in range(len(block_ids))
        ]
        expected = [bytes([65 + index]) * block_size for index in range(len(block_ids))]
        for block, payload in zip(blocks, expected):
            block[:] = payload
        offsets = [0] * len(block_ids)
        lengths = [block_size] * len(block_ids)
        nsids = [nsid] * len(block_ids)

        assert pynds_kv.init(storage, len(storage)) == 0
        assert pynds_kv.init_multi([storage], [len(storage)]) == 0
        assert pynds_kv.batchPut(block_ids, blocks, offsets, lengths, nsids) == 0
        assert pynds_kv.is_exists(block_ids, nsid) == 2
        assert pynds_kv.is_exists(block_ids, nsid + 1) == 0
        assert pynds_kv.is_exists([1001, 9999, 1002], nsid) == 1

        for block in blocks:
            block[:] = bytes(block_size)
        assert pynds_kv.batchGet(block_ids, blocks, offsets, lengths, nsids) == 0
        assert [block.tobytes() for block in blocks] == expected

    print("Mock read/write and 4 KiB alignment example passed")


if __name__ == "__main__":
    main()
