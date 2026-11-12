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
#ifndef ATB_KVCACHE_PREFETCH_H
#define ATB_KVCACHE_PREFETCH_H
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <omp.h>
#include <pybind11/numpy.h>
#include <queue>
#include <sstream>
#include <stdarg.h>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "ucmstore_v1.h"

namespace py = pybind11;
namespace ucmprefetch {
using BlockId = std::array<std::byte, 16>; /* 16-byte block hash */

class __attribute__((visibility("hidden"))) UCMKVPrefetch {
private:
    UC::StoreV1* mStoreV1_{nullptr};
    uint32_t mLayerNum = 78;

    enum class PrefetchStatus : int32_t {
        GSA_PREFETCH_DONE = 0,
        GSA_STORE_EMPTY,
        GSA_BLOCKID_EMPTY,
        GSA_HANDLE_ERROR,
        GSA_LOOKUP_ERROR
    };

public:
    UCMKVPrefetch() {}

    ~UCMKVPrefetch() {}

    void StoreInit(uintptr_t storePtr, uint32_t layerNum)
    {
        mStoreV1_ = reinterpret_cast<UC::StoreV1*>(storePtr);
        if (mStoreV1_ == nullptr) { throw std::runtime_error("storeV1 is null"); }
        mLayerNum = layerNum;
    }

    int32_t UCMLoadSSD(const std::vector<UC::Detail::BlockId>& blockIds)
    {
        if (mStoreV1_ == nullptr) { return static_cast<int32_t>(PrefetchStatus::GSA_STORE_EMPTY); }

        if (blockIds.empty()) { return static_cast<int32_t>(PrefetchStatus::GSA_BLOCKID_EMPTY); }
        UC::Detail::TaskDesc task;
        task.reserve(blockIds.size() * mLayerNum);

        for (const auto& blockId : blockIds) {
            for (uint32_t layerID = 0; layerID < mLayerNum; layerID++) {
                task.emplace_back(UC::Detail::Shard{blockId, layerID, {}});
            }
        }
        auto loadHandle = mStoreV1_->Load(std::move(task));
        if (!loadHandle) { return static_cast<int32_t>(PrefetchStatus::GSA_HANDLE_ERROR); }
        return static_cast<int32_t>(PrefetchStatus::GSA_PREFETCH_DONE);
    }

    std::vector<uint8_t> UCMLookup(const std::vector<UC::Detail::BlockId>& blockIds)
    {
        if (mStoreV1_ == nullptr) { return {}; }

        if (blockIds.empty()) { return {}; }

        auto lookupResult = mStoreV1_->Lookup(blockIds.data(), blockIds.size());
        if (!lookupResult) { return {}; }

        auto result = lookupResult.Value();
        if (result.size() != blockIds.size()) { return {}; }
        return result;
    }
};

}  // namespace ucmprefetch

#endif