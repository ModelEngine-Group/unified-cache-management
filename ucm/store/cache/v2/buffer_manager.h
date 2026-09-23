/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#pragma once

#include <cstddef>
#include <sys/types.h>
#include <vector>
#include "cache_buffer.h"
#include "metrics_api.h"
#include "time/stopwatch.h"
#include "ucmstore_v1.h"

namespace UC::Cache2 {

template <typename BufferT = Buffer>
class BufferManager {
    BufferT buffer_;
    StoreV1* backend_{nullptr};

public:
    Status Setup(const Config& config)
    {
        backend_ = config.storeBackend;
        return buffer_.Setup(config);
    }
    BufferT* GetTransBuffer() { return &buffer_; }
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num)
    {
        std::vector<Detail::BlockId> missBlocks;
        std::vector<size_t> missIdx;
        missBlocks.reserve(num);
        missIdx.reserve(num);
        StopWatch sw;
        for (size_t i = 0; i < num; ++i) {
            if (!buffer_.Exist(blocks[i], 0)) {
                missBlocks.push_back(blocks[i]);
                missIdx.push_back(i);
            }
        }
        ReportLookupStats(sw, num - missBlocks.size(), missBlocks.size());
        if (missBlocks.empty()) { return static_cast<ssize_t>(num) - 1; }
        if (!backend_) { return static_cast<ssize_t>(missIdx[0]) - 1; }
        StopWatch backendSw;
        auto res = backend_->LookupOnPrefix(missBlocks.data(), missBlocks.size());
        if (!res) [[unlikely]] { return res.Error(); }
        ReportBackendLookupStats(backendSw);
        const auto found = res.Value();
        if (static_cast<size_t>(found + 1) == missIdx.size()) {
            return static_cast<ssize_t>(num) - 1;
        }
        return static_cast<ssize_t>(missIdx[found + 1]) - 1;
    }
    Expected<ssize_t> LookupOnReverse(const Detail::BlockId* blocks, size_t num)
    {
        StopWatch sw;
        ssize_t bufferHitIdx = -1;
        for (ssize_t i = static_cast<ssize_t>(num) - 1; i >= 0; --i) {
            if (buffer_.Exist(blocks[i], 0)) {
                bufferHitIdx = i;
                break;
            }
        }
        const size_t hits = static_cast<size_t>(bufferHitIdx + 1);
        ReportLookupStats(sw, hits, num - hits);
        if (bufferHitIdx == static_cast<ssize_t>(num) - 1) { return bufferHitIdx; }
        if (!backend_) { return bufferHitIdx; }
        const std::vector<Detail::BlockId> backendMiss(blocks + (bufferHitIdx + 1), blocks + num);
        StopWatch backendSw;
        auto res = backend_->LookupOnReverse(backendMiss.data(), backendMiss.size());
        if (!res) [[unlikely]] { return res.Error(); }
        ReportBackendLookupStats(backendSw);
        const auto found = res.Value();
        if (found < 0) { return bufferHitIdx; }
        return bufferHitIdx + 1 + static_cast<ssize_t>(found);
    }
    void Prefetch(const Detail::BlockId* blocks, size_t num)
    {
        buffer_.Touch(blocks, num);
        if (backend_) { backend_->Prefetch(blocks, num); }
    }

private:
    void ReportLookupStats(const StopWatch& sw, size_t hits, size_t misses)
    {
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_lookup_duration_ms"),
                             sw.Elapsed().count() * 1e3);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_lookup_hit_blocks_total"),
                             static_cast<double>(hits));
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_lookup_miss_blocks_total"),
                             static_cast<double>(misses));
    }
    void ReportBackendLookupStats(const StopWatch& sw)
    {
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_lookup_backend_duration_ms"),
                             sw.Elapsed().count() * 1e3);
    }
};

}  // namespace UC::Cache2
