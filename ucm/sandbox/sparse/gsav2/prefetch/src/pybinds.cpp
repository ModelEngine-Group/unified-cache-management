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

std::uintptr_t GetTensorDataPtr(const pybind11::object& tensor, const char* tensorName)
{
    if (tensor.is_none()) {
        throw pybind11::value_error(std::string(tensorName) + " must not be None");
    }

    if (!pybind11::hasattr(tensor, "data_ptr")) {
        throw pybind11::type_error(std::string(tensorName) + " must be a torch.Tensor");
    }

    if (!pybind11::hasattr(tensor, "dim") || !pybind11::hasattr(tensor, "numel") ||
        !pybind11::hasattr(tensor, "is_contiguous")) {
        throw pybind11::type_error(std::string(tensorName) +
                                   " does not provide required Tensor interfaces");
    }

    const std::int64_t ndim = tensor.attr("dim")().cast<std::int64_t>();
    if (ndim != 1) {
        throw pybind11::value_error(std::string(tensorName) + " must be a one-dimensional Tensor");
    }

    const std::int64_t numel = tensor.attr("numel")().cast<std::int64_t>();
    if (numel <= 0) { throw pybind11::value_error(std::string(tensorName) + " must not be empty"); }

    const bool is_contiguous = tensor.attr("is_contiguous")().cast<bool>();
    if (!is_contiguous) {
        throw pybind11::value_error(std::string(tensorName) + " must be contiguous");
    }

    const std::uintptr_t address = tensor.attr("data_ptr")().cast<std::uintptr_t>();
    if (address == 0) {
        throw pybind11::value_error(std::string(tensorName) + " data_ptr is null");
    }

    return address;
}

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

int32_t UCMLoadSSDFromBuffer(UCMKVPrefetch& ucmPrefetch, const pybind11::buffer& ids)
{
    std::vector<BlockId> blocksHash = BlockIdsFromBuffer(ids);
    return ucmPrefetch.UCMLoadSSD(blocksHash);
}

std::vector<uint8_t> UCMLookupFromBuffer(UCMKVPrefetch& ucmPrefetch, const pybind11::buffer& ids)
{
    std::vector<BlockId> blocksHash = BlockIdsFromBuffer(ids);
    return ucmPrefetch.UCMLookup(blocksHash);
}

PYBIND11_MODULE(gsa_prefetch, m)
{
    pybind11::class_<ucmprefetch::UCMKVPrefetch>(m, "UCMKVPrefetch")
        .def(pybind11::init<>())
        .def("store_init", &ucmprefetch::UCMKVPrefetch::StoreInit, pybind11::arg("store_ptr"),
             pybind11::arg("layer_num"))
        .def("ucm_load_ssd", &ucmprefetch::UCMLoadSSDFromBuffer,
             pybind11::arg("block_ids").noconvert())
        .def("ucm_lookup", &ucmprefetch::UCMLookupFromBuffer,
             pybind11::arg("block_ids").noconvert());

    pybind11::class_<ucmprefetch::GSAPrefetchEngineC>(m, "GSAPrefetchEngineC")
        .def(pybind11::init<std::vector<int>&, std::vector<int>&, bool, int, int, std::vector<int>,
                            std::vector<int>, std::string>())
        // NOTE: get_self_ptr / set_store_v1 pass raw pointers across the Python
        // boundary as uintptr_t. The caller MUST guarantee that the engine
        // (get_self_ptr) and the UCM store (set_store_v1) outlive every use on
        // the Python side: any use after the C++ object/store is destroyed is a
        // dangling-pointer use-after-free. A pybind11 owning wrapper that holds
        // the pointer is the long-term fix.
        .def("set_blocks_map", &ucmprefetch::SetBlocksMapFromBuffer, pybind11::arg("req_id"),
             pybind11::arg("ids").noconvert(), pybind11::arg("max_idx"),
             pybind11::arg("block_idxs"), pybind11::arg("k_addrs"), pybind11::arg("v_addrs"))
        .def("set_last_indexer_block", &ucmprefetch::GSAPrefetchEngineC::SetLastIndexerBlock,
             pybind11::arg("req_id"), pybind11::arg("block_idxs"), pybind11::arg("indexer_addrs"))
        .def("set_tokens_map", &ucmprefetch::GSAPrefetchEngineC::SetTokensMap)
        .def("del_blocks_map", &ucmprefetch::GSAPrefetchEngineC::DelBlocksMap)
        .def("set_topk_indices", &ucmprefetch::GSAPrefetchEngineC::SetTopkIndices)
        .def("pre_swap_tokens", &ucmprefetch::GSAPrefetchEngineC::PreSwapTokens)
        .def("submit_prefetch_task", &ucmprefetch::GSAPrefetchEngineC::SubmitTask)
        .def("set_kvTensor_Layer", &ucmprefetch::GSAPrefetchEngineC::SetKvTensor)
        .def("set_kvTensor_Index", &ucmprefetch::GSAPrefetchEngineC::SetKvTensorIndex)
        .def("set_kv_buffer", &ucmprefetch::GSAPrefetchEngineC::SetKvBuffer)
        .def("get_prefetch_status", &ucmprefetch::GSAPrefetchEngineC::GetPrefetchStatus)
        .def("set_modelrunning_status", &ucmprefetch::GSAPrefetchEngineC::SetModelRunningStatus)
        .def("get_token_map", &ucmprefetch::GSAPrefetchEngineC::GetTokenBlockMap)
        .def("get_block_map", &ucmprefetch::GSAPrefetchEngineC::GetBlockTokenMap)
        .def("get_block_table", &ucmprefetch::GSAPrefetchEngineC::GetReqUseBlockTable)
        .def("get_sparse_len", &ucmprefetch::GSAPrefetchEngineC::GetSparseSeqLen)
        .def("set_indexer_map", &ucmprefetch::GSAPrefetchEngineC::SetIndexerMap)
        .def("update_indexer_map", &ucmprefetch::GSAPrefetchEngineC::UpdateIndexerMap)
        .def("submit_task_indexer", &ucmprefetch::GSAPrefetchEngineC::SubmitTaskIndexer)
        .def("get_indexer_status", &ucmprefetch::GSAPrefetchEngineC::GetIndexerStatus)
        .def("indexer_init_load", &ucmprefetch::GSAPrefetchEngineC::IndexerInitLoad)
        .def("indexer_block_map", &ucmprefetch::GSAPrefetchEngineC::GetIndexerBlockMap)
        .def("indexer_token_map", &ucmprefetch::GSAPrefetchEngineC::GetIndexerTokensMap)
        .def("set_store_v1", &ucmprefetch::GSAPrefetchEngineC::SetStoreV1Raw)
        .def("get_store_v1_ptr", &ucmprefetch::GSAPrefetchEngineC::GetStoreV1Raw)
        .def("get_self_ptr", &ucmprefetch::GSAPrefetchEngineC::GetSelfRaw);
}
}  // namespace ucmprefetch