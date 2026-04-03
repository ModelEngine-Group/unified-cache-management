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
#include <numeric>
#include "buffer_manager.h"
#include "logger/logger.h"
#include "trans_manager.h"

namespace UC::CacheStore {

class CacheStore : public StoreV1 {
    BufferManager bufferMgr_;
    bool transEnable_{false};
    TransManager transMgr_;

public:
    Status Setup(const Detail::Dictionary& inConfig) override
    {
        auto config = ParseConfig(inConfig);
        auto s = CheckConfig(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed to check config params: {}.", s);
            return s;
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
    void Prefetch(const Detail::BlockId* blocks, size_t num) override {}
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
        if (s.Failure()) [[unlikely]] { UC_ERROR("Failed({}) to wait task({}).", s, taskId); }
        return s;
    }

private:
    Config ParseConfig(const Detail::Dictionary& config)
    {
        Config param;
        config.Get("store_backend", param.storeBackend);
        config.Get("unique_id", param.uniqueId);
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
        return param;
    }
    Status CheckSizeConfig(const Config& config)
    {
        if (config.tensorSizes.empty()) { return Status::InvalidParam("invalid tensor size"); }
        if (config.shardSize == 0) { return Status::InvalidParam("invalid shard size"); }
        if (config.blockSize == 0) { return Status::InvalidParam("invalid block size"); }
        if (std::accumulate(config.tensorSizes.begin(), config.tensorSizes.end(), size_t(0)) !=
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
        if (config.uniqueId.empty()) { return Status::InvalidParam("invalid unique id"); }
        for (const auto core : config.cpuAffinityCores) {
            if (core < 0 || core >= CPU_SETSIZE) {
                return Status::InvalidParam("invalid cpu core({})", core);
            }
        }
        if (config.deviceId == -1) { return Status::OK(); }
        auto s = CheckSizeConfig(config);
        if (s.Failure()) { return s; }
        auto bufferNumber = config.bufferCapacity / config.shardSize;
        if (bufferNumber < 1024 || bufferNumber < config.loadExclusiveBufferNumber * 2) {
            return Status::InvalidParam("too small buffer({}) on shard({})", config.bufferCapacity,
                                        config.shardSize);
        }
        if (config.waitingQueueDepth <= 1 || config.runningQueueDepth <= 1) {
            return Status::InvalidParam("invalid queue depth({},{})", config.waitingQueueDepth,
                                        config.runningQueueDepth);
        }
        if (config.streamNumber < 1 || config.streamNumber > 32) {
            return Status::InvalidParam("invalid stream number({})", config.streamNumber);
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
        UC_INFO("Set {}::WaitingQueueDepth to {}.", ns, config.waitingQueueDepth);
        UC_INFO("Set {}::RunningQueueDepth to {}.", ns, config.runningQueueDepth);
        UC_INFO("Set {}::TimeoutMs to {}.", ns, config.timeoutMs);
        UC_INFO("Set {}::StreamNumber to {}.", ns, config.streamNumber);
    }
};

}  // namespace UC::CacheStore

extern "C" UC::StoreV1* MakeCacheStore() { return new UC::CacheStore::CacheStore(); }
