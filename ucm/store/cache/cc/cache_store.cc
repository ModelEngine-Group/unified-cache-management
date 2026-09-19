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
#include <algorithm>
#include <memory>
#include <numeric>
#include "buffer_manager.h"
#include "logger/logger.h"
#include "trans/cuda/gdr/gdr_config.h"
#include "trans_manager.h"

#ifndef UCM_RUNTIME_ASCEND_IO_AGGREGATION
#define UCM_RUNTIME_ASCEND_IO_AGGREGATION 0
#endif

#ifndef UCM_RUNTIME_ASCEND_SDMA_DIRECT
#define UCM_RUNTIME_ASCEND_SDMA_DIRECT 0
#endif

namespace UC::CacheStore {

class CacheStore : public StoreV1 {
    BufferManager bufferMgr_;
    bool transEnable_{false};
    TransManager transMgr_;
    std::unique_ptr<Trans::GdrKVBufferConfig> gpuKvBufferRegistrations_{nullptr};

public:
    Status Setup(const Detail::Dictionary& inConfig) override
    {
        auto config = ParseConfig(inConfig);
        auto s = CheckConfig(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed to check config params: {}.", s);
            return s;
        }
        if (!config.cacheUseHostBuffer && config.deviceId >= 0 &&
            !config.gpuKvBufferAddrs.empty()) {
            gpuKvBufferRegistrations_ = std::make_unique<Trans::GdrKVBufferConfig>();
            s = gpuKvBufferRegistrations_->Register(config.gpuKvBufferAddrs,
                                                    config.gpuKvBufferSizes);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to register GPU KV buffers.", s);
                return s;
            }
        }
        s = bufferMgr_.Setup(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup buffer manager.", s);
            return s;
        }
        transEnable_ = config.deviceId >= 0;
        if (transEnable_) {
            s = transMgr_.Setup(config, bufferMgr_.GetTransBuffer());
            if (s.Failure()) [[unlikely]] { return s; }
        }
        ShowConfig(config);
        return Status::OK();
    }
    std::string Readme() const override { return "CacheStore"; }
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override
    {
        auto res = bufferMgr_.Lookup(blocks, num);
        if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
        return res;
    }
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override
    {
        auto res = bufferMgr_.LookupOnPrefix(blocks, num);
        if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
        return res;
    }
    Expected<ssize_t> LookupOnReverse(const Detail::BlockId* blocks, size_t num) override
    {
        auto res = bufferMgr_.LookupOnReverse(blocks, num);
        if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
        return res;
    }
    void Prefetch(const Detail::BlockId* blocks, size_t num) override
    {
        bufferMgr_.Prefetch(blocks, num);
    }
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        if (!transEnable_) { return Status::Error("transfer is not enable"); }
        auto res = transMgr_.Submit({TransTask::Type::LOAD, std::move(task)});
        if (!res) [[unlikely]] {
            UC_ERROR("Failed({}) to submit load task({}).", res.Error(), task.brief);
        }
        return res;
    }
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        if (!transEnable_) { return Status::Error("transfer is not enable"); }
        auto res = transMgr_.Submit({TransTask::Type::DUMP, std::move(task)});
        if (!res) [[unlikely]] {
            UC_ERROR("Failed({}) to submit dump task({}).", res.Error(), task.brief);
        }
        return res;
    }
    Expected<bool> Check(Detail::TaskHandle taskId) override
    {
        auto res = transMgr_.Check(taskId);
        if (!res) [[unlikely]] { UC_ERROR("Failed({}) to check task({}).", res.Error(), taskId); }
        return res;
    }
    Status Wait(Detail::TaskHandle taskId) override
    {
        auto s = transMgr_.Wait(taskId);
        if (s.Failure() && s != Status::StoreUnhealthy()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait task({}).", s, taskId);
        }
        return s;
    }

private:
    Config ParseConfig(const Detail::Dictionary& config)
    {
        Config param;
        config.Get("store_backend", param.storeBackend);
        config.Get("unique_id", param.uniqueId);
        config.Get("cache_load_backend_only", param.cacheLoadBackendOnly);
        config.GetNumber("device_id", param.deviceId);
        size_t tensorSize = 0;
        config.GetNumber("tensor_size", tensorSize);
        config.GetNumber("shard_size", param.shardSize);
        if (tensorSize != 0) {
            param.tensorSizes.assign(param.shardSize / tensorSize, tensorSize);
        } else {
            config.GetNumbers("tensor_size_list", param.tensorSizes);
        }
        config.GetNumber("block_size", param.blockSize);
        config.Get("cpu_affinity_cores", param.cpuAffinityCores);
        if (param.shardSize > 0) { param.waitingQueueDepth *= (param.blockSize / param.shardSize); }
        config.Get("share_buffer_enable", param.shareBufferEnable);
        if (!param.shareBufferEnable) { param.bufferCapacity /= 8; }
        config.Get("io_direct", param.ioDirect);
        size_t bufferCapacityGb = 0;
        config.GetNumber("cache_buffer_capacity_gb", bufferCapacityGb);
        if (bufferCapacityGb != 0) { param.bufferCapacity = bufferCapacityGb << 30; }
        config.GetNumber("waiting_queue_depth", param.waitingQueueDepth);
        config.GetNumber("running_queue_depth", param.runningQueueDepth);
        config.GetNumber("timeout_ms", param.timeoutMs);
        config.GetNumber("cache_stream_number", param.streamNumber);
        config.GetNumber("cache_load_exclusive_buffer_number", param.loadExclusiveBufferNumber);
        config.GetNumbers("gpu_kv_buffer_addrs", param.gpuKvBufferAddrs);
        config.GetNumbers("gpu_kv_buffer_sizes", param.gpuKvBufferSizes);
        config.Get("use_gdr", param.useGdr);
        config.Get("cache_io_aggregation", param.cacheIOAggregation);
        param.cacheIOAggregation = param.cacheIOAggregation && UCM_RUNTIME_ASCEND_IO_AGGREGATION;
        config.Get("cache_sdma_direct", param.cacheSdmaDirect);
        config.Get("cache_use_host_buffer", param.cacheUseHostBuffer);
        config.GetNumber("cache_h2h_worker_number", param.h2hWorkerNumber);
        config.GetNumber("cache_h2h_queue_depth", param.h2hQueueDepth);
        config.GetNumber("local_rank_size", param.localRankSize);
        return param;
    }
    Status CheckSizeConfig(const Config& config)
    {
        if (config.tensorSizes.empty()) { return Status::InvalidParam("invalid tensor size"); }
        if (config.shardSize == 0) { return Status::InvalidParam("invalid shard size"); }
        if (config.blockSize == 0) { return Status::InvalidParam("invalid block size"); }
        if (std::accumulate(config.tensorSizes.begin(), config.tensorSizes.end(), size_t(0)) >
            config.shardSize) {
            return Status::InvalidParam("invalid shard size({})", config.shardSize);
        }
        if (config.blockSize % config.shardSize != 0) {
            return Status::InvalidParam("invalid block size({})", config.blockSize);
        }
        return Status::OK();
    }
    Status CheckConfig(const Config& config)
    {
        if (!config.storeBackend) { return Status::InvalidParam("invalid store backend"); }
        if (config.deviceId < -1) {
            return Status::InvalidParam("invalid device({})", config.deviceId);
        }
        if (config.cacheUseHostBuffer && config.deviceId < 0) {
            return Status::InvalidParam(
                "cache_use_host_buffer requires a non-negative device_id as cache owner id");
        }
        if (config.uniqueId.empty()) { return Status::InvalidParam("invalid unique id"); }
        auto s =
            Trans::GdrKVBufferConfig::Validate(config.gpuKvBufferAddrs, config.gpuKvBufferSizes);
        if (s.Failure()) { return s; }
        for (const auto core : config.cpuAffinityCores) {
            if (core < 0 || core >= CPU_SETSIZE) {
                return Status::InvalidParam("invalid cpu core({})", core);
            }
        }
        if (config.deviceId == -1) { return Status::OK(); }
        s = CheckSizeConfig(config);
        if (s.Failure()) { return s; }
        if (config.cacheUseHostBuffer &&
            (config.cacheIOAggregation || config.cacheSdmaDirect || config.useGdr ||
             !config.gpuKvBufferAddrs.empty())) {
            return Status::InvalidParam(
                "cache_use_host_buffer is incompatible with cache_io_aggregation, "
                "cache_sdma_direct, use_gdr, and gpu_kv_buffer_addrs");
        }
        if (config.cacheUseHostBuffer &&
            (config.h2hWorkerNumber == 0 || config.h2hQueueDepth == 0)) {
            return Status::InvalidParam("invalid H2H worker/queue configuration({},{})",
                                        config.h2hWorkerNumber, config.h2hQueueDepth);
        }
#if !UCM_RUNTIME_ASCEND_SDMA_DIRECT
        if (config.cacheSdmaDirect) {
            return Status::InvalidParam("Cache SDMA Direct requires RUNTIME_ENVIRONMENT=ascend-a3");
        }
#endif
        auto bufferNumber = config.bufferCapacity / config.shardSize;
        const size_t minBufferNumber = std::max(size_t(1024), config.loadExclusiveBufferNumber * 2);
        if (bufferNumber < minBufferNumber) {
            const size_t minBufferCapacityGb =
                (minBufferNumber * config.shardSize + (size_t(1) << 30) - 1) >> 30;
            return Status::InvalidParam(
                "too small buffer({}) on shard({}), please set cache_buffer_capacity_gb >= {}GB",
                config.bufferCapacity, config.shardSize, minBufferCapacityGb);
        }
        if (config.waitingQueueDepth <= 1 || config.runningQueueDepth <= 1) {
            return Status::InvalidParam("invalid queue depth({},{})", config.waitingQueueDepth,
                                        config.runningQueueDepth);
        }
        if (config.cacheIOAggregation && config.cacheSdmaDirect) {
            return Status::InvalidParam(
                "Cache IO aggregation is incompatible with Cache SDMA Direct");
        }
        if (config.streamNumber < 1 || config.streamNumber > 32) {
            return Status::InvalidParam("invalid stream number({})", config.streamNumber);
        }
        if (config.localRankSize == 0) {
            return Status::InvalidParam("invalid local rank size({})", config.localRankSize);
        }
        return Status::OK();
    }
    void ShowConfig(const Config& config)
    {
        constexpr const char* ns = "CacheStore";
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
        UC_INFO("Set {}::StoreBackend to {}.", ns, config.storeBackend->Readme());
        UC_INFO("Set {}::UniqueId to {}.", ns, config.uniqueId);
        UC_INFO("Set {}::CacheLoadBackendOnly to {}.", ns, config.cacheLoadBackendOnly);
        UC_INFO("Set {}::DeviceId to {}.", ns, config.deviceId);
        const auto& v = config.tensorSizes;
        if (v.empty()) {
            UC_INFO("Set {}::TensorSizes to [].", ns);
        } else if (std::all_of(v.begin(), v.end(), [&](auto d) { return d == v[0]; })) {
            UC_INFO("Set {}::TensorSizes to {}(*{}).", ns, v[0], v.size());
        } else {
            UC_INFO("Set {}::TensorSizes to {}.", ns, v);
        }
        UC_INFO("Set {}::ShardSize to {}.", ns, config.shardSize);
        UC_INFO("Set {}::BlockSize to {}.", ns, config.blockSize);
        UC_INFO("Set {}::IoDirect to {}.", ns, config.ioDirect);
        UC_INFO("Set {}::CpuAffinityCores to {}.", ns, config.cpuAffinityCores);
        UC_INFO("Set {}::BufferCapacity to {}GB.", ns, config.bufferCapacity >> 30);
        UC_INFO("Set {}::ShareBufferEnable to {}.", ns, config.shareBufferEnable);
        UC_INFO("Set {}::CacheIOAggregation to {}.", ns, config.cacheIOAggregation);
        if (config.cacheIOAggregation) {
            UC_INFO("Set {}::AggregationObject to CacheStoreShard.", ns);
        }
        UC_INFO("Set {}::WaitingQueueDepth to {}.", ns, config.waitingQueueDepth);
        UC_INFO("Set {}::RunningQueueDepth to {}.", ns, config.runningQueueDepth);
        UC_INFO("Set {}::TimeoutMs to {}.", ns, config.timeoutMs);
        if (config.cacheSdmaDirect) {
            UC_INFO(
                "Set {}::StreamNumber to {} (configured={}, Cache SDMA Direct uses one stream).",
                ns, config.EffectiveStreamNumber(), config.streamNumber);
        } else {
            UC_INFO("Set {}::StreamNumber to {}.", ns, config.EffectiveStreamNumber());
        }
        UC_INFO("Set {}::CacheSdmaDirect to {}.", ns, config.cacheSdmaDirect);
        UC_INFO("Set {}::CacheUseHostBuffer to {}.", ns, config.cacheUseHostBuffer);
        UC_INFO("Set {}::H2HWorkerNumber to {}.", ns, config.h2hWorkerNumber);
        UC_INFO("Set {}::H2HQueueDepth to {}.", ns, config.h2hQueueDepth);
        UC_INFO("Set {}::LoadExclusiveBufferNumber to {}.", ns, config.loadExclusiveBufferNumber);
        UC_INFO("Set {}::GpuKvBufferNumber to {}.", ns, config.gpuKvBufferAddrs.size());
        UC_INFO("Set {}::UseGdr to {}.", ns, config.useGdr);
        UC_INFO("Set {}::LocalRankSize to {}.", ns, config.localRankSize);
    }
};

}  // namespace UC::CacheStore

extern "C" UC::StoreV1* MakeCacheStore() { return new UC::CacheStore::CacheStore(); }
