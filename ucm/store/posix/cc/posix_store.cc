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
#include "posix_store.h"
#include <fmt/ranges.h>
#include "hotness_tracker.h"
#include "logger/logger.h"
#include "shard_gc.h"
#include "space_manager.h"
#include "trans_manager.h"

namespace UC::PosixStore {

class PosixStoreImpl {
public:
    SpaceManager spaceMgr;
    TransManager transMgr;
    ShardGarbageCollector gcMgr;
    HotnessTracker hotnessTracker;
    bool transEnable{false};
    bool gcEnable{false};

public:
    Status Setup(const Config& config)
    {
        auto s = CheckConfig(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed to check config params: {}.", s);
            return s;
        }
        transEnable = config.deviceId >= 0;

        bool isScheduler = (config.deviceId == -1);
        bool isDP0 = false;
        if (!config.uniqueId.empty() && config.uniqueId.size() >= 4) {
            size_t dpPos = config.uniqueId.rfind("_dp");
            if (dpPos != std::string::npos && dpPos == config.uniqueId.size() - 4) {
                isDP0 = (config.uniqueId.back() == '0');
            }
        }

        bool isDPScenario = (config.uniqueId.find("_dp") != std::string::npos);
        gcEnable = config.posixStorageGcEnable && isScheduler && (!isDPScenario || isDP0);

        ShowConfig(config);

        if (isDPScenario) {
            UC_DEBUG("DP scenario, uniqueId={}, isDP0={}, gcEnable={}", config.uniqueId, isDP0,
                     gcEnable);
        } else {
            UC_DEBUG("Non-DP scenario, isScheduler={}, gcEnable={}", isScheduler, gcEnable);
        }

        if (gcEnable) {
            s = hotnessTracker.Setup(spaceMgr.GetLayout(), config.hotnessInterval);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed to setup HotnessTracker: {}.", s);
                return s;
            }
        }
        s = spaceMgr.Setup(config, gcEnable ? &hotnessTracker : nullptr);
        if (s.Failure()) [[unlikely]] { return s; }
        if (gcEnable) {
            ShardGCConfig gcConfig;
            gcConfig.recyclePercent = config.gcRecyclePercent;
            gcConfig.gcConcurrency = config.gcConcurrency;
            s = gcMgr.Setup(spaceMgr.GetLayout(), config.storageBackends, gcConfig);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed to setup GC: {}.", s);
                return s;
            }
            if (config.posixStorageCapacityGb > 0) {
                size_t storageCapacityBytes =
                    config.posixStorageCapacityGb * 1024ULL * 1024ULL * 1024ULL;
                size_t maxFileCount = storageCapacityBytes / config.blockSize;
                auto shards = spaceMgr.GetLayout()->RelativeRoots();
                size_t totalShards = shards.size();
                if (totalShards > 0) {
                    UC_INFO("GC enabled: capacityGb={}, thresholdFilesPerShard={}",
                            config.posixStorageCapacityGb,
                            static_cast<size_t>(maxFileCount / totalShards *
                                                config.gcTriggerThresholdRatio));
                }
                gcMgr.SetGCThreshold(maxFileCount, config.gcTriggerThresholdRatio);
                hotnessTracker.SetGCTrigger(&gcMgr, maxFileCount, config.gcTriggerThresholdRatio);
                if (gcMgr.ShouldTrigger(maxFileCount, config.gcTriggerThresholdRatio)) {
                    gcMgr.Trigger();
                }
            }
        }
        if (transEnable) {
            s = transMgr.Setup(config, spaceMgr.GetLayout());
            if (s.Failure()) [[unlikely]] { return s; }
        }
        return Status::OK();
    }

private:
    Status CheckConfig(const Config& config)
    {
        if (config.storageBackends.empty()) {
            return Status::InvalidParam("invalid storage backends");
        }
        if (config.deviceId < -1) {
            return Status::InvalidParam("invalid device({})", config.deviceId);
        }
        if (config.dataTransConcurrency == 0 || config.lookupConcurrency == 0) {
            return Status::InvalidParam("invalid concurrency({},{})", config.dataTransConcurrency,
                                        config.lookupConcurrency);
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
        return Status::OK();
    }
    void ShowConfig(const Config& config)
    {
        constexpr const char* ns = "PosixStore";
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
        UC_INFO("Set {}::StorageBackends to {}.", ns, config.storageBackends);
        UC_INFO("Set {}::UniqueId to {}.", ns, config.uniqueId);
        UC_INFO("Set {}::DeviceId to {}.", ns, config.deviceId);
        UC_INFO("Set {}::TensorSize to {}.", ns, config.tensorSize);
        UC_INFO("Set {}::ShardSize to {}.", ns, config.shardSize);
        UC_INFO("Set {}::BlockSize to {}.", ns, config.blockSize);
        UC_INFO("Set {}::IoDirect to {}.", ns, config.ioDirect);
        UC_INFO("Set {}::DataTransConcurrency to {}.", ns, config.dataTransConcurrency);
        UC_INFO("Set {}::LookupConcurrency to {}.", ns, config.lookupConcurrency);
        UC_INFO("Set {}::TimeoutMs to {}.", ns, config.timeoutMs);
        UC_INFO("Set {}::DataDirShardBytes to {}.", ns, config.dataDirShardBytes);
    }
};

PosixStore::PosixStore::~PosixStore() = default;

Status PosixStore::Setup(const Detail::Dictionary& config)
{
    Config param;
    config.Get("storage_backends", param.storageBackends);
    config.Get("unique_id", param.uniqueId);
    config.GetNumber("device_id", param.deviceId);
    config.GetNumber("tensor_size", param.tensorSize);
    config.GetNumber("shard_size", param.shardSize);
    config.GetNumber("block_size", param.blockSize);
    config.Get("io_direct", param.ioDirect);
    config.GetNumber("posix_data_trans_concurrency", param.dataTransConcurrency);
    config.GetNumber("posix_lookup_concurrency", param.lookupConcurrency);
    config.GetNumber("timeout_ms", param.timeoutMs);
    config.GetNumber("data_dir_shard_bytes", param.dataDirShardBytes);
    config.Get("posix_storage_gc_enable", param.posixStorageGcEnable);
    config.Get("gc_recycle_percent", param.gcRecyclePercent);
    config.GetNumber("gc_concurrency", param.gcConcurrency);
    config.GetNumber("hotness_interval", param.hotnessInterval);
    config.GetNumber("posix_storage_capacity_gb", param.posixStorageCapacityGb);
    config.Get("gc_trigger_threshold_ratio", param.gcTriggerThresholdRatio);
    try {
        impl_ = std::make_shared<PosixStoreImpl>();
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make posix store object.", e.what());
        return Status::Error(e.what());
    }
    return impl_->Setup(param);
}

std::string PosixStore::Readme() const { return "PosixStore"; }

Expected<std::vector<uint8_t>> PosixStore::PosixStore::Lookup(const Detail::BlockId* blocks,
                                                              size_t num)
{
    auto res = impl_->spaceMgr.Lookup(blocks, num);
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
    return res;
}

Expected<ssize_t> PosixStore::LookupOnPrefix(const Detail::BlockId* blocks, size_t num)
{
    auto res = impl_->spaceMgr.LookupOnPrefix(blocks, num);
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
    return res;
}

void PosixStore::PosixStore::Prefetch(const Detail::BlockId* blocks, size_t num) {}

Expected<Detail::TaskHandle> PosixStore::PosixStore::Load(Detail::TaskDesc task)
{
    if (!impl_->transEnable) { return Status::Error("transfer is not enable"); }
    auto res = impl_->transMgr.Submit({TransTask::Type::LOAD, std::move(task)});
    if (!res) [[unlikely]] {
        UC_ERROR("Failed({}) to submit load task({}).", res.Error(), task.brief);
    }
    return res;
}

Expected<Detail::TaskHandle> PosixStore::PosixStore::Dump(Detail::TaskDesc task)
{
    if (!impl_->transEnable) { return Status::Error("transfer is not enable"); }
    auto res = impl_->transMgr.Submit({TransTask::Type::DUMP, std::move(task)});
    if (!res) [[unlikely]] {
        UC_ERROR("Failed({}) to submit dump task({}).", res.Error(), task.brief);
    }
    return res;
}

Expected<bool> PosixStore::PosixStore::Check(Detail::TaskHandle taskId)
{
    auto res = impl_->transMgr.Check(taskId);
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to check task({}).", res.Error(), taskId); }
    return res;
}

Status PosixStore::PosixStore::Wait(Detail::TaskHandle taskId)
{
    auto s = impl_->transMgr.Wait(taskId);
    if (s.Failure()) [[unlikely]] { UC_ERROR("Failed({}) to wait task({}).", s, taskId); }
    return s;
}

void PosixStore::NotifyAccess(const Detail::BlockId* blocks, size_t num)
{
    if (!impl_->gcEnable) { return; }
    for (size_t i = 0; i < num; ++i) { impl_->hotnessTracker.Touch(blocks[i]); }
}

void PosixStore::TriggerGC()
{
    if (!impl_->gcEnable) { return; }
    impl_->gcMgr.Trigger();
}

Expected<size_t> PosixStore::ExecuteGC()
{
    if (!impl_->gcEnable) { return Status::Error("GC is not enabled"); }
    return impl_->gcMgr.Execute();
}

bool PosixStore::IsGCRunning() const
{
    if (!impl_->gcEnable) { return false; }
    return impl_->gcMgr.IsRunning();
}

}  // namespace UC::PosixStore

extern "C" UC::StoreV1* MakePosixStore() { return new UC::PosixStore::PosixStore(); }
