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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>
#include "logger/logger.h"
#include "status/status.h"
#include "type/dictionary.h"
#include "ucmstore_v1.h"

namespace UC::Cache2 {

struct Config {
    StoreV1* storeBackend{};
    std::string uniqueId{};
    int32_t deviceId{-1};
    std::vector<size_t> tensorSizes{};
    size_t shardSize{0};
    size_t blockSize{0};
    size_t bufferCapacity{256ULL << 30};
    size_t loadExclusiveBufferNumber{1024};
    size_t waitingQueueDepth{8192};
    size_t runningQueueDepth{524288};
    size_t timeoutMs{30000};
    size_t streamNumber{4};
    size_t localRankSize{8};

    static Config From(const Detail::Dictionary& dict)
    {
        Config config;
        dict.Get("store_backend", config.storeBackend);
        dict.Get("unique_id", config.uniqueId);
        dict.GetNumber("device_id", config.deviceId);
        size_t tensorSize = 0;
        dict.GetNumber("tensor_size", tensorSize);
        dict.GetNumber("shard_size", config.shardSize);
        if (tensorSize != 0) {
            config.tensorSizes.assign(config.shardSize / tensorSize, tensorSize);
        } else {
            dict.GetNumbers("tensor_size_list", config.tensorSizes);
        }
        dict.GetNumber("block_size", config.blockSize);
        size_t bufferCapacityGb = 0;
        dict.GetNumber("cache_buffer_capacity_gb", bufferCapacityGb);
        if (bufferCapacityGb != 0) { config.bufferCapacity = bufferCapacityGb << 30; }
        dict.GetNumber("cache_load_exclusive_buffer_number", config.loadExclusiveBufferNumber);
        dict.GetNumber("waiting_queue_depth", config.waitingQueueDepth);
        dict.GetNumber("running_queue_depth", config.runningQueueDepth);
        dict.GetNumber("timeout_ms", config.timeoutMs);
        dict.GetNumber("cache_stream_number", config.streamNumber);
        dict.GetNumber("local_rank_size", config.localRankSize);
        return config;
    }
    Status Validate() const
    {
        if (deviceId < -1) { return Status::InvalidParam("invalid device({})", deviceId); }
        if (uniqueId.empty()) { return Status::InvalidParam("invalid unique id"); }
        if (deviceId == -1) { return Status::OK(); }
        if (tensorSizes.empty()) { return Status::InvalidParam("invalid tensor size"); }
        if (shardSize == 0) { return Status::InvalidParam("invalid shard size"); }
        if (blockSize == 0) { return Status::InvalidParam("invalid block size"); }
        if (std::accumulate(tensorSizes.begin(), tensorSizes.end(), size_t(0)) > shardSize) {
            return Status::InvalidParam("invalid shard size({})", shardSize);
        }
        if (blockSize % shardSize != 0) {
            return Status::InvalidParam("invalid block size({})", blockSize);
        }
        const auto bufferNumber = bufferCapacity / shardSize;
        const size_t minBufferNumber = std::max(size_t(1024), loadExclusiveBufferNumber * 2);
        if (bufferNumber < minBufferNumber) {
            const size_t minBufferCapacityGb =
                (minBufferNumber * shardSize + (size_t(1) << 30) - 1) >> 30;
            return Status::InvalidParam(
                "too small buffer({}) on shard({}), please set cache_buffer_capacity_gb >= {}GB",
                bufferCapacity, shardSize, minBufferCapacityGb);
        }
        if (waitingQueueDepth <= 1 || runningQueueDepth <= 1) {
            return Status::InvalidParam("invalid queue depth({},{})", waitingQueueDepth,
                                        runningQueueDepth);
        }
        if (streamNumber < 1 || streamNumber > 32) {
            return Status::InvalidParam("invalid stream number({})", streamNumber);
        }
        if (localRankSize == 0) {
            return Status::InvalidParam("invalid local rank size({})", localRankSize);
        }
        return Status::OK();
    }
    void Show() const
    {
        constexpr const char* ns = "CacheStore";
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
        UC_INFO("Set {}::StoreBackend to {}.", ns,
                storeBackend ? storeBackend->Readme() : "nullptr");
        UC_INFO("Set {}::UniqueId to {}.", ns, uniqueId);
        UC_INFO("Set {}::DeviceId to {}.", ns, deviceId);
        const auto& v = tensorSizes;
        if (v.empty()) {
            UC_INFO("Set {}::TensorSizes to [].", ns);
        } else if (std::all_of(v.begin(), v.end(), [&](auto d) { return d == v[0]; })) {
            UC_INFO("Set {}::TensorSizes to {}(*{}).", ns, v[0], v.size());
        } else {
            UC_INFO("Set {}::TensorSizes to {}.", ns, v);
        }
        UC_INFO("Set {}::ShardSize to {}.", ns, shardSize);
        UC_INFO("Set {}::BlockSize to {}.", ns, blockSize);
        UC_INFO("Set {}::BufferCapacity to {}GB.", ns, bufferCapacity >> 30);
        UC_INFO("Set {}::LoadExclusiveBufferNumber to {}.", ns, loadExclusiveBufferNumber);
        UC_INFO("Set {}::WaitingQueueDepth to {}.", ns, waitingQueueDepth);
        UC_INFO("Set {}::RunningQueueDepth to {}.", ns, runningQueueDepth);
        UC_INFO("Set {}::TimeoutMs to {}.", ns, timeoutMs);
        UC_INFO("Set {}::StreamNumber to {}.", ns, streamNumber);
        UC_INFO("Set {}::LocalRankSize to {}.", ns, localRankSize);
    }
};

}  // namespace UC::Cache2
