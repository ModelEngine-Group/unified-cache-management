import time
from collections import defaultdict

import numpy as np
import torch

from ucm.sparse.gsa_on_device.hash_encoder import HashEncoder
from ucm.sparse.gsa_on_device.hash_retrieval import hash_retrieval_backend
from ucm.sparse.kvstar.utils import get_bind_cpus_for_rank


class HashRetrievalWorker:
    # handle torch -> numpy && float16/bfloat16 -> float32.
    def __init__(self, cpp_worker):
        self.cpp_worker = cpp_worker

    @classmethod
    def handle_input(cls, input):
        if input.dtype != torch.uint8:
            input = input.to(torch.uint8)
        input = input.to("cpu", non_blocking=True)
        return input

    def submit(self, query, topk, indexes):
        q = self.handle_input(query)
        req_id = self.cpp_worker.submit(q, topk, indexes)
        return req_id

    def poll(self, req_id):
        return self.cpp_worker.poll(req_id)  # Returns True if ready

    def get_result(self, req_id):
        return self.cpp_worker.get_result(req_id)

    def wait(self, req_id):
        return self.cpp_worker.wait(req_id)


if __name__ == "__main__":
    ################# data
    batch_size = 2
    block_size = 2
    head_dim = 128
    head_num = 1
    dim = head_dim * head_num
    kv_cache_blocks = 2560
    data = torch.rand(kv_cache_blocks, block_size, dim).to(torch.float32)
    print("data created", data.shape)

    topk = 10
    search_blocks_range = 100
    tpot = 30 / 1000

    indexes = np.arange(batch_size * search_blocks_range).reshape(
        batch_size, search_blocks_range
    )

    query = torch.rand(batch_size, dim).to(torch.float32)

    hash_encoder = HashEncoder(
        input_dim=dim,
        hash_bits=dim,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    hash_query = hash_encoder.compute_hash(query)
    hash_key_cache = hash_encoder.compute_hash(data)

    ratio = 0.75
    total_tp_size = 4
    local_tp_rank = 0
    bind_info_list, alloc_numa_ids = get_bind_cpus_for_rank(
        total_tp_size, local_tp_rank, ratio=ratio
    )

    bind_info_dict = defaultdict(list)
    for item in bind_info_list:
        bind_info_dict[item[1]].append(item[0])
    bind_info_dict = dict(bind_info_dict)

    backend = hash_retrieval_backend.HashRetrievalWorkerBackend(
        hash_key_cache, bind_info_dict
    )
    worker = HashRetrievalWorker(backend)

    #################### cpp async version
    req_id = worker.submit(hash_query, topk=topk, indexes=indexes)

    #################### LLM decode begin
    time.sleep(tpot * 3)
    #################### LLM decode done

    # Poll and get result (in a real program, you'd likely use asyncio or threading)
    begin = time.time()
    worker.wait(req_id)
    result = worker.get_result(req_id)
    print("cpp spent:", time.time() - begin)
    cpp_indices = np.sort(result["indices"], 1)
    print(f"cpp indices={cpp_indices}")

    ################### numpy version
    unpacked_hash_query = hash_encoder._unpack_hash(hash_query)
    unpacked_hash_key_cache = hash_encoder._unpack_hash(hash_key_cache)
    begin = time.time()
    data_indexed = unpacked_hash_key_cache[indexes.flatten()].reshape(
        indexes.shape[0], indexes.shape[1], block_size, dim
    )
    scores = torch.einsum("td, tnjd->tnj", unpacked_hash_query, data_indexed)

    block_scores_ret = torch.max(scores, dim=-1)
    blocks_scores = block_scores_ret.values

    topk_ret = torch.topk(blocks_scores, topk, dim=-1)
    topk_index = topk_ret.indices
    topk_index = topk_index.sort(dim=-1).values
    topk_index = indexes[np.arange(indexes.shape[0])[:, None], topk_index]
    print("numpy spent: ", time.time() - begin)
    print(f"numpy indices={topk_index}")
