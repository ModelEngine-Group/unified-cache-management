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
#include "ds3fs_store.h"
#include <fmt/ranges.h>
#include "logger/logger.h"
#include "space_manager.h"
#include "trans_manager.h"

namespace UC::Ds3fsStore {

class Ds3fsStoreImpl {
public:
    SpaceManager spaceMgr;
    TransManager transMgr;
    bool transEnable{false};

public:
    Status Setup(const Config& config)
    {
        auto s = CheckConfig(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed to check config params: {}.", s);
            return s;
        }
        s = spaceMgr.Setup(config);
        if (s.Failure()) [[unlikely]] { return s; }
        transEnable = config.deviceId >= 0;
        if (transEnable) {
            s = transMgr.Setup(config, spaceMgr.GetLayout());
            if (s.Failure()) [[unlikely]] { return s; }
        }
        ShowConfig(config);
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
        if (config.deviceId == -1) { return Status::OK(); }
        if (config.tensorSize == 0 || config.shardSize < config.tensorSize ||
            config.blockSize < config.shardSize || config.shardSize % config.tensorSize != 0 ||
            config.blockSize % config.shardSize != 0) {
            return Status::InvalidParam("invalid size({},{},{})", config.tensorSize,
                                        config.shardSize, config.blockSize);
        }
        if (config.streamNumber == 0) {
            return Status::InvalidParam("invalid stream number({})", config.streamNumber);
        }
        return Status::OK();
    }
    void ShowConfig(const Config& config)
    {
        constexpr const char* ns = "Ds3fsStore";
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
        UC_INFO("Set {}::StorageBackends to {}.", ns, config.storageBackends[0]);
        UC_INFO("Set {}::DeviceId to {}.", ns, config.deviceId);
        if (config.deviceId == -1) { return; }
        UC_INFO("Set {}::TensorSize to {}.", ns, config.tensorSize);
        UC_INFO("Set {}::ShardSize to {}.", ns, config.shardSize);
        UC_INFO("Set {}::BlockSize to {}.", ns, config.blockSize);
        UC_INFO("Set {}::IoDirect to {}.", ns, config.ioDirect);
        UC_INFO("Set {}::StreamNumber to {}.", ns, config.streamNumber);
        UC_INFO("Set {}::TimeoutMs to {}.", ns, config.timeoutMs);
    }
};

Ds3fsStore::~Ds3fsStore() = default;

Status Ds3fsStore::Setup(const Detail::Dictionary& config)
{
    Config param;
    config.Get("storage_backends", param.storageBackends);
    config.Get("device_id", param.deviceId);
    config.Get("tensor_size", param.tensorSize);
    config.Get("shard_size", param.shardSize);
    config.Get("block_size", param.blockSize);
    config.Get("io_direct", param.ioDirect);
    config.Get("stream_number", param.streamNumber);
    config.Get("timeout_ms", param.timeoutMs);
    config.Get("ior_entries", param.iorEntries);
    config.Get("ior_depth", param.iorDepth);
    config.Get("numa_id", param.numaId);
    try {
        impl_ = std::make_shared<Ds3fsStoreImpl>();
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make ds3fs store object.", e.what());
        return Status::Error(e.what());
    }
    return impl_->Setup(param);
}

std::string Ds3fsStore::Readme() const { return "Ds3fsStore"; }

Expected<std::vector<uint8_t>> Ds3fsStore::Lookup(const Detail::BlockId* blocks, size_t num)
{
    return impl_->spaceMgr.Lookup(blocks, num);
}

void Ds3fsStore::Prefetch(const Detail::BlockId* blocks, size_t num) {}

Expected<Detail::TaskHandle> Ds3fsStore::Load(Detail::TaskDesc task)
{
    if (!impl_->transEnable) { return Status::Error("transfer is not enable"); }
    auto res = impl_->transMgr.Submit({TransTask::Type::LOAD, std::move(task)});
    if (!res) [[unlikely]] {
        UC_ERROR("Failed({}) to submit load task({}).", res.Error(), task.brief);
    }
    return res;
}

Expected<Detail::TaskHandle> Ds3fsStore::Dump(Detail::TaskDesc task)
{
    if (!impl_->transEnable) { return Status::Error("transfer is not enable"); }
    auto res = impl_->transMgr.Submit({TransTask::Type::DUMP, std::move(task)});
    if (!res) [[unlikely]] {
        UC_ERROR("Failed({}) to submit dump task({}).", res.Error(), task.brief);
    }
    return res;
}

Expected<bool> Ds3fsStore::Check(Detail::TaskHandle taskId)
{
    auto res = impl_->transMgr.Check(taskId);
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to check task({}).", res.Error(), taskId); }
    return res;
}

Status Ds3fsStore::Wait(Detail::TaskHandle taskId)
{
    auto s = impl_->transMgr.Wait(taskId);
    if (s.Failure()) [[unlikely]] { UC_ERROR("Failed({}) to wait task({}).", s, taskId); }
    return s;
}

}  // namespace UC::Ds3fsStore

extern "C" UC::StoreV1* MakeDs3fsStore() { return new UC::Ds3fsStore::Ds3fsStore(); }
