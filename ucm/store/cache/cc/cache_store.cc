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
#include "cache_store.h"
#include <shared_mutex>
#include "load_queue.h"
#include "logger/logger.h"
#include "template/hashset.h"
#include "trans_manager.h"

namespace UC::CacheStore {

class CacheStoreImpl {
public:
    StoreV1* backend{nullptr};
    bool transEnable{false};
    TransManager transMgr;

public:
    Status Setup(const Config& config)
    {
        auto s = CheckConfig(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed to check config params: {}.", s);
            return s;
        }
        backend = static_cast<StoreV1*>((void*)config.backend);
        transEnable = config.deviceId >= 0;
        if (transEnable) {
            s = transMgr.Setup(config);
            if (s.Failure()) [[unlikely]] { return s; }
        }
        ShowConfig(config);
        return Status::OK();
    }

private:
    Status CheckConfig(const Config& config)
    {
        if (!config.backend) { return Status::InvalidParam("invalid backend"); }
        if (config.deviceId < -1) {
            return Status::InvalidParam("invalid device({})", config.deviceId);
        }
        if (config.engineId.empty()) { return Status::InvalidParam("invalid engine id"); }
        if (config.deviceId == -1) { return Status::OK(); }
        if (config.tensorSize == 0 || config.shardSize < config.tensorSize ||
            config.blockSize < config.shardSize || config.shardSize % config.tensorSize != 0 ||
            config.blockSize % config.shardSize != 0) {
            return Status::InvalidParam("invalid size({},{},{})", config.tensorSize,
                                        config.shardSize, config.blockSize);
        }
        if (config.bufferSize < config.blockSize * 1024) {
            return Status::InvalidParam("too small buffer size({})", config.bufferSize);
        }
        if (config.bufferSize % config.shardSize != 0) {
            return Status::InvalidParam("invalid buffer size({})", config.bufferSize);
        }
        if (config.waitingQueueDepth <= 1 || config.runningQueueDepth <= 1) {
            return Status::InvalidParam("invalid queue depth({},{})", config.waitingQueueDepth,
                                        config.runningQueueDepth);
        }
        return Status::OK();
    }
    void ShowConfig(const Config& config)
    {
        constexpr const char* ns = "CacheStore";
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
        UC_INFO("Set {}::EngineId to {}.", ns, config.engineId);
        UC_INFO("Set {}::DeviceId to {}.", ns, config.deviceId);
        if (config.deviceId < 0) { return; }
        UC_INFO("Set {}::TensorSize to {}.", ns, config.tensorSize);
        UC_INFO("Set {}::ShardSize to {}.", ns, config.shardSize);
        UC_INFO("Set {}::BlockSize to {}.", ns, config.blockSize);
        UC_INFO("Set {}::BufferSize to {}.", ns, config.bufferSize);
        UC_INFO("Set {}::ShareBufferEnable to {}.", ns, config.shareBufferEnable);
        UC_INFO("Set {}::WaitingQueueDepth to {}.", ns, config.waitingQueueDepth);
        UC_INFO("Set {}::RunningQueueDepth to {}.", ns, config.runningQueueDepth);
        UC_INFO("Set {}::TransferTimeoutMs to {}.", ns, config.transferTimeoutMs);
    }
};

CacheStore::~CacheStore() = default;

Status CacheStore::Setup(const Config& config)
{
    try {
        impl_ = std::make_shared<CacheStoreImpl>();
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make cache store object.", e.what());
        return Status::Error(e.what());
    }
    return impl_->Setup(config);
}

Expected<std::vector<uint8_t>> CacheStore::Lookup(const Detail::BlockId* blocks, size_t num)
{
    auto res = impl_->backend->Lookup(blocks, num);
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
    return res;
}

void CacheStore::Prefetch(const Detail::BlockId*, size_t) {}

Expected<Detail::TaskHandle> CacheStore::Load(Detail::TaskDesc task)
{
    if (!impl_->transEnable) { return Status::Error("transfer is not enable"); }
    auto res = impl_->transMgr.Submit({TransTask::Type::LOAD, std::move(task)});
    if (!res) [[unlikely]] {
        UC_ERROR("Failed({}) to submit load task({}).", res.Error(), task.brief);
    }
    return res;
}

Expected<Detail::TaskHandle> CacheStore::Dump(Detail::TaskDesc task)
{
    if (!impl_->transEnable) { return Status::Error("transfer is not enable"); }
    auto res = impl_->transMgr.Submit({TransTask::Type::DUMP, std::move(task)});
    if (!res) [[unlikely]] {
        UC_ERROR("Failed({}) to submit dump task({}).", res.Error(), task.brief);
    }
    return res;
}

Expected<bool> CacheStore::Check(Detail::TaskHandle taskId)
{
    auto res = impl_->transMgr.Check(taskId);
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to check task({}).", res.Error(), taskId); }
    return res;
}

Status CacheStore::Wait(Detail::TaskHandle taskId)
{
    auto s = impl_->transMgr.Wait(taskId);
    if (s.Failure()) [[unlikely]] { UC_ERROR("Failed({}) to wait task({}).", s, taskId); }
    return s;
}

}  // namespace UC::CacheStore
