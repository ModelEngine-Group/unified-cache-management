/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include <cstdint>
#include <cstring>
#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <torch/extension.h>
#include <vector>
#include "kvcache_pre.h"
#include "kvcache_prefetch.h"

namespace ucmprefetch {

std::vector<BlockId> BlockIdsFromBuffer(const pybind11::buffer& ids)
{
    pybind11::buffer_info ids_info = ids.request();
    if (ids_info.ndim != 1) { throw pybind11::value_error("ids must be a one-dimensional buffer"); }

    if (ids_info.itemsize != sizeof(std::uint8_t)) {
        throw pybind11::value_error("ids item size must be 1 byte");
    }

    if (ids_info.format != pybind11::format_descriptor<std::uint8_t>::format()) {
        throw pybind11::value_error("ids dtype must be uint8");
    }

    if (ids_info.strides.empty() || ids_info.strides[0] != ids_info.itemsize) {
        throw pybind11::value_error("ids must be contiguous");
    }
    const std::size_t byteSize =
        static_cast<std::size_t>(ids_info.size) * static_cast<std::size_t>(ids_info.itemsize);
    if (byteSize % sizeof(BlockId) != 0) {
        throw pybind11::value_error("ids byte size must be a multiple of BlockId size");
    }
    const std::size_t blockCount = byteSize / sizeof(BlockId);
    std::vector<BlockId> blocksIds(blockCount);
    const auto* source = static_cast<const std::uint8_t*>(ids_info.ptr);
    for (std::size_t i = 0; i < blockCount; ++i) {
        std::memcpy(blocksIds[i].data(), source + i * sizeof(BlockId), sizeof(BlockId));
    }
    return blocksIds;
}

void SetBlocksMapFromBuffer(GSAPrefetchEngineC& engine, const std::string& reqID,
                            const pybind11::buffer& ids, int maxIdx,
                            const std::vector<int>& blockIdxs,
                            const std::vector<std::uintptr_t>& kAddrs,
                            const std::vector<std::uintptr_t>& vAddrs)
{
    std::vector<BlockId> blocksHash = BlockIdsFromBuffer(ids);
    engine.SetBlocksMap(reqID, blocksHash, maxIdx, blockIdxs, kAddrs, vAddrs);
}

int64_t UCMLoadSSDFromBuffer(UCMKVPrefetch& ucmPrefetch, const pybind11::buffer& ids)
{
    std::vector<BlockId> blocksHash = BlockIdsFromBuffer(ids);
    return ucmPrefetch.UCMLoadSSD(blocksHash);
}

int32_t UCMCheckTask(UCMKVPrefetch& ucmPrefetch, std::uint64_t taskId)
{
    return ucmPrefetch.UCMCheck(taskId);
}

int32_t UCMWaitTask(UCMKVPrefetch& ucmPrefetch, std::uint64_t taskId)
{
    return ucmPrefetch.UCMWait(taskId);
}

std::vector<uint8_t> UCMLookupFromBuffer(UCMKVPrefetch& ucmPrefetch, const pybind11::buffer& ids)
{
    std::vector<BlockId> blocksHash = BlockIdsFromBuffer(ids);
    return ucmPrefetch.UCMLookup(blocksHash);
}

namespace {

// UCMKVPrefetch 绑定。
void registerUcmKvPrefetch(pybind11::module_& m)
{
    pybind11::class_<ucmprefetch::UCMKVPrefetch>(m, "UCMKVPrefetch")
        .def(pybind11::init<>())
        .def("store_init", &ucmprefetch::UCMKVPrefetch::StoreInit, pybind11::arg("store_ptr"),
             pybind11::arg("layer_num"), pybind11::arg("tensor_count"))
        .def("ucm_load_ssd", &ucmprefetch::UCMLoadSSDFromBuffer,
             pybind11::arg("block_ids").noconvert())
        .def("ucm_check", &ucmprefetch::UCMCheckTask, pybind11::arg("task_id"))
        .def("ucm_wait", &ucmprefetch::UCMWaitTask, pybind11::arg("task_id"),
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("ucm_lookup", &ucmprefetch::UCMLookupFromBuffer,
             pybind11::arg("block_ids").noconvert());
}

// GSAPrefetchEngineC 绑定。
void registerPrefetchEngine(pybind11::module_& m)
{
    pybind11::class_<ucmprefetch::GSAPrefetchEngineC>(m, "GSAPrefetchEngineC")
        .def(pybind11::init<std::vector<int>&, std::vector<int>&, bool, int, int, std::vector<int>,
                            std::vector<int>, std::string>())
        // NOTE: get_self_ptr / set_store_v1 pass raw pointers across the Python
        // boundary as uintptr_t: the engine and the UCM store MUST outlive every
        // Python-side use (else use-after-free); owning wrapper is the long-term fix.
        .def("set_blocks_map", &ucmprefetch::SetBlocksMapFromBuffer, pybind11::arg("req_id"),
             pybind11::arg("ids").noconvert(), pybind11::arg("max_idx"),
             pybind11::arg("block_idxs"), pybind11::arg("k_addrs"), pybind11::arg("v_addrs"))
        .def("set_last_indexer_block", &ucmprefetch::GSAPrefetchEngineC::SetLastIndexerBlock,
             pybind11::arg("req_id"), pybind11::arg("block_idxs"), pybind11::arg("indexer_addrs"))
        .def("set_tokens_map", &ucmprefetch::GSAPrefetchEngineC::SetTokensMap)
        // HBM 驱逐模式：mode 1 = decay（全部层启用衰减驱逐 + Budget5120 分层窗口），
        // 其他值 = lru（默认，与原实现一致）。需在首个 set_tokens_map 之前调用。
        .def("set_hbm_evict_mode", &ucmprefetch::GSAPrefetchEngineC::SetHbmEvictMode,
             pybind11::arg("mode"))
        .def("get_hbm_evict_mode", &ucmprefetch::GSAPrefetchEngineC::GetHbmEvictMode)
        .def("del_blocks_map", &ucmprefetch::GSAPrefetchEngineC::DelBlocksMap)
        .def("set_topk_indices", &ucmprefetch::GSAPrefetchEngineC::SetTopkIndices)
        .def("pre_swap_tokens", &ucmprefetch::GSAPrefetchEngineC::PreSwapTokens)
        .def("submit_prefetch_task", &ucmprefetch::GSAPrefetchEngineC::SubmitTask)
        .def("set_kvTensor_Layer", &ucmprefetch::GSAPrefetchEngineC::SetKvTensor)
        .def("set_kvTensor_Index", &ucmprefetch::GSAPrefetchEngineC::SetKvTensorIndex)
        .def("set_kv_buffer", &ucmprefetch::GSAPrefetchEngineC::SetKvBuffer)
        .def("get_prefetch_status", &ucmprefetch::GSAPrefetchEngineC::GetPrefetchStatus)
        .def("set_indexer_map", &ucmprefetch::GSAPrefetchEngineC::SetIndexerMap)
        .def("update_indexer_map", &ucmprefetch::GSAPrefetchEngineC::UpdateIndexerMap)
        .def("submit_task_indexer", &ucmprefetch::GSAPrefetchEngineC::SubmitTaskIndexer)
        .def("get_indexer_status", &ucmprefetch::GSAPrefetchEngineC::GetIndexerStatus)
        // 事件等待必须 gil_scoped_release：pybind 默认持 GIL 进 C++，阻塞等待
        // 会卡死所有 Python 线程（time.sleep 本身是释放 GIL 的）。
        .def("wait_prefetch_idle", &ucmprefetch::GSAPrefetchEngineC::WaitPrefetchIdle,
             pybind11::arg("timeout_ms"), pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("wait_indexer_idle", &ucmprefetch::GSAPrefetchEngineC::WaitIndexerIdle,
             pybind11::arg("timeout_ms"), pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("indexer_init_load", &ucmprefetch::GSAPrefetchEngineC::IndexerInitLoad)
        .def("set_store_v1", &ucmprefetch::GSAPrefetchEngineC::SetStoreV1Raw)
        .def("get_store_v1_ptr", &ucmprefetch::GSAPrefetchEngineC::GetStoreV1Raw)
        .def("set_store_tensor_count", &ucmprefetch::GSAPrefetchEngineC::SetStoreTensorCount)
        .def("get_store_tensor_count", &ucmprefetch::GSAPrefetchEngineC::GetStoreTensorCount)
        .def("get_self_ptr", &ucmprefetch::GSAPrefetchEngineC::GetSelfRaw);
}

}  // namespace

PYBIND11_MODULE(gsa_prefetch, m)
{
    registerUcmKvPrefetch(m);
    registerPrefetchEngine(m);
}
}  // namespace ucmprefetch