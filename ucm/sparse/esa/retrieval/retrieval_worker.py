import time
from collections import defaultdict

import numpy as np
import torch

from ucm.sparse.esa.retrieval import retrieval_backend
from ucm.sparse.kvstar.utils import get_bind_cpus_for_rank


class RetrievalWorker:
    # handle torch -> numpy && float16/bfloat16 -> float32.
    def __init__(self, cpp_worker):
        self.cpp_worker = cpp_worker

    @classmethod
    def handle_input(cls, input):
        if input.dtype != torch.float32:
            input = input.to(torch.float32)
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
    dim = 1024
    kv_cache_blocks = 25600
    data = torch.rand(kv_cache_blocks, dim).to(torch.float32)
    print("data created", data.shape)

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

    backend = retrieval_backend.RetrievalWorkerBackend(data, bind_info_dict)
    worker = RetrievalWorker(backend)
    topk = 3000
    search_blocks_range = 8000
    tpot = 30 / 1000

    indexes = np.arange(batch_size * search_blocks_range).reshape(
        batch_size, search_blocks_range
    )

    query = torch.rand(batch_size, dim).to(torch.float32)

    #################### cpp async version
    req_id = worker.submit(query, topk=topk, indexes=indexes)

    #################### LLM decode begin
    time.sleep(tpot * 3)
    #################### LLM decode done

    # Poll and get result (in a real program, you'd likely use asyncio or threading)
    begin = time.time()
    worker.wait(req_id)
    result = worker.get_result(req_id)
    print("cpp spent:", time.time() - begin)

    ################### numpy version
    begin = time.time()
    data_indexed = (
        data[indexes.flatten()].reshape(indexes.shape[0], indexes.shape[1], dim).numpy()
    )
    query = RetrievalWorker.handle_input(query)
    scores = np.matmul(query[:, None, :], data_indexed.transpose((0, 2, 1)))
    scores = scores[:, 0, :]
    topk_elements = np.partition(scores, -topk, -1)[:, -topk:]
    topk_indices = np.argpartition(scores, -topk, -1)[:, -topk:]
    topk_indices = indexes[np.arange(indexes.shape[0])[:, None], topk_indices]
    print("numpy spent: ", time.time() - begin)

    ## compare
    cpp_elements = np.sort(result["scores"], 1)
    cpp_indices = np.sort(result["indices"], 1)

    np_elements = np.sort(topk_elements, 1)
    np_indices = np.sort(topk_indices, 1)

    diff_elements = np.abs(np_elements - cpp_elements)
    diff_indices = np.abs(np_indices - cpp_indices)

    print(f"diff topk: {diff_indices.max()}")
