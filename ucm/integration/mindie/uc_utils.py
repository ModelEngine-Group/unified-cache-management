import numpy as np


class KVPtrComputer:
    def __init__(self, npu_cache):
        self.num_layers = len(npu_cache)
        assert self.num_layers > 0, "empty npu_cache"

        ref = npu_cache[0][0]
        self.max_num_blocks = int(ref.shape[0])

        for layer in range(self.num_layers):
            for kv in (0, 1):
                t = npu_cache[layer][kv]
                assert (
                    int(t.shape[0]) == self.max_num_blocks
                ), f"dim0 mismatch at layer={layer}, k or v={kv}: {t.shape[0]} vs {self.max_num_blocks}"

        self.ptr_table = np.empty(
            (self.num_layers, 2, self.max_num_blocks), dtype=np.uint64
        )
        ar = np.arange(self.max_num_blocks, dtype=np.uint64)

        for layer in range(self.num_layers):
            for kv in (0, 1):
                t = npu_cache[layer][kv]
                elem = t.element_size()
                stor_base = t.untyped_storage().data_ptr()
                base = np.uint64(stor_base + t.storage_offset() * elem)
                stride0 = np.uint64(t.stride(0) * elem)
                self.ptr_table[layer, kv, :] = base + stride0 * ar

    def ptrs_for_blocks_np(self, block_ids):
        idx = np.asarray(block_ids, dtype=np.intp)
        return self.ptr_table[:, :, idx].transpose(2, 0, 1)
