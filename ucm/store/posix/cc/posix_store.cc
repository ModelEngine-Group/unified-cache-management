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
#include <fmt/ranges.h>
#include "logger/logger.h"
#include "space_manager.h"
#include "trans_manager.h"
#include "ucmstore_v1.h"

namespace UC::PosixStore {

class PosixStore : public StoreV1 {
    SpaceManager spaceMgr_;
    TransManager transMgr_;
    bool transEnable_{false};

public:
    Status Setup(const Detail::Dictionary& inConfig) override
    {
        auto config = ParseConfig(inConfig);
        auto s = CheckConfig(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed to check config params: {}.", s);
            return s;
        }
        transEnable_ = config.deviceId >= 0;

        ShowConfig(config);

        s = spaceMgr_.Setup(config);
        if (s.Failure()) [[unlikely]] { return s; }

        if (transEnable_) {
            s = transMgr_.Setup(config, spaceMgr_.GetLayout());
            if (s.Failure()) [[unlikely]] { return s; }
        }
        return Status::OK();
    }
    std::string Readme() const override { return "PosixStore"; }
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override
    {
        auto res = spaceMgr_.Lookup(blocks, num);
        if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
        return res;
    }
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override
    {
        auto res = spaceMgr_.LookupOnPrefix(blocks, num);
        if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
        return res;
    }
    void Prefetch(const Detail::BlockId* blocks, size_t num) override {}
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        if (!transEnable_) { return Status::Error("transfer is not enable"); }
        auto res = transMgr_.GetIoEngine()->Submit({TransTask::Type::LOAD, std::move(task)});
        if (!res) [[unlikely]] {
            UC_ERROR("Failed({}) to submit load task({}).", res.Error(), task.brief);
        }
        return res;
    }
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        if (!transEnable_) { return Status::Error("transfer is not enable"); }
        auto res = transMgr_.GetIoEngine()->Submit({TransTask::Type::DUMP, std::move(task)});
        if (!res) [[unlikely]] {
            UC_ERROR("Failed({}) to submit dump task({}).", res.Error(), task.brief);
        }
        return res;
    }
    Expected<bool> Check(Detail::TaskHandle taskId) override
    {
        auto res = transMgr_.GetIoEngine()->Check(taskId);
        if (!res) [[unlikely]] { UC_ERROR("Failed({}) to check task({}).", res.Error(), taskId); }
        return res;
    }
    Status Wait(Detail::TaskHandle taskId) override
    {
        auto s = transMgr_.GetIoEngine()->Wait(taskId);
        if (s.Failure()) [[unlikely]] { UC_ERROR("Failed({}) to wait task({}).", s, taskId); }
        return s;
    }

private:
    Config ParseConfig(const Detail::Dictionary& inConfig)
    {
        Config config;
        inConfig.Get("storage_backends", config.storageBackends);
        inConfig.GetNumber("device_id", config.deviceId);
        inConfig.GetNumber("tensor_size", config.tensorSize);
        inConfig.GetNumber("shard_size", config.shardSize);
        inConfig.GetNumber("block_size", config.blockSize);
        inConfig.Get("posix_io_engine", config.ioEngine);
        inConfig.Get("io_direct", config.ioDirect);
        inConfig.GetNumber("posix_data_trans_concurrency", config.dataTransConcurrency);
        inConfig.GetNumber("posix_lookup_concurrency", config.lookupConcurrency);
        inConfig.GetNumber("posix_open_concurrency", config.openConcurrency);
        inConfig.GetNumber("posix_commit_concurrency", config.commitConcurrency);
        inConfig.GetNumber("timeout_ms", config.timeoutMs);
        inConfig.GetNumber("data_dir_shard_bytes", config.dataDirShardBytes);
        inConfig.Get("posix_gc_enable", config.posixGcEnable);
        inConfig.Get("posix_gc_recycle_percent", config.posixGcRecyclePercent);
        inConfig.GetNumber("posix_gc_concurrency", config.posixGcConcurrency);
        inConfig.GetNumber("posix_gc_check_interval_sec", config.posixGcCheckIntervalSec);
        inConfig.GetNumber("posix_capacity_gb", config.posixCapacityGb);
        inConfig.Get("posix_gc_trigger_threshold_ratio", config.posixGcTriggerThresholdRatio);
        inConfig.GetNumber("posix_gc_max_recycle_count_per_shard",
                           config.posixGcMaxRecycleCountPerShard);
        inConfig.Get("posix_gc_shard_sample_ratio", config.posixGcShardSampleRatio);
        return config;
    }
    Status CheckConfig(const Config& config)
    {
        if (config.storageBackends.empty()) {
            return Status::InvalidParam("invalid storage backends");
        }
        if (config.deviceId < -1) {
            return Status::InvalidParam("invalid device({})", config.deviceId);
        }
        if (config.lookupConcurrency == 0) {
            return Status::InvalidParam("invalid lookup concurrency({})", config.lookupConcurrency);
        }
        if (config.dataDirShardBytes > 5) {
            return Status::InvalidParam("invalid shard bytes({})", config.dataDirShardBytes);
        }
        if (config.deviceId == -1) { return Status::OK(); }
        if (config.tensorSize == 0 || config.shardSize < config.tensorSize ||
            config.blockSize < config.shardSize || config.shardSize % config.tensorSize != 0 ||
            config.blockSize % config.shardSize != 0) {
            return Status::InvalidParam("invalid size({},{},{})", config.tensorSize,
                                        config.shardSize, config.blockSize);
        }
        if (config.ioEngine == "aio") {
            if (config.openConcurrency == 0 || config.commitConcurrency == 0) {
                return Status::InvalidParam("invalid aio concurrency({},{})",
                                            config.openConcurrency, config.commitConcurrency);
            }
        } else if (config.ioEngine == "psync") {
            if (config.dataTransConcurrency == 0) {
                return Status::InvalidParam("invalid psync concurrency({})",
                                            config.dataTransConcurrency);
            }
        } else {
            return Status::InvalidParam("invalid io engine({})", config.ioEngine);
        }
        return Status::OK();
    }
    void ShowConfig(const Config& config)
    {
        constexpr const char* ns = "PosixStore";
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
        UC_INFO("Set {}::StorageBackends to {}.", ns, config.storageBackends);
        UC_INFO("Set {}::DeviceId to {}.", ns, config.deviceId);
        UC_INFO("Set {}::TensorSize to {}.", ns, config.tensorSize);
        UC_INFO("Set {}::ShardSize to {}.", ns, config.shardSize);
        UC_INFO("Set {}::BlockSize to {}.", ns, config.blockSize);
        UC_INFO("Set {}::IoEngine to {}.", ns, config.ioEngine);
        UC_INFO("Set {}::IoDirect to {}.", ns, config.ioDirect);
        UC_INFO("Set {}::DataTransConcurrency to {}.", ns, config.dataTransConcurrency);
        UC_INFO("Set {}::LookupConcurrency to {}.", ns, config.lookupConcurrency);
        UC_INFO("Set {}::OpenConcurrency to {}.", ns, config.openConcurrency);
        UC_INFO("Set {}::CommitConcurrency to {}.", ns, config.commitConcurrency);
        UC_INFO("Set {}::TimeoutMs to {}.", ns, config.timeoutMs);
        UC_INFO("Set {}::DataDirShardBytes to {}.", ns, config.dataDirShardBytes);
    }
};

}  // namespace UC::PosixStore

extern "C" UC::StoreV1* MakePosixStore() { return new UC::PosixStore::PosixStore(); }
