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
#include "logger/logger.h"
#include "space_manager.h"
#include "trans_manager.h"

namespace UC::PosixStore {

class PosixStoreImpl {
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
        transEnable = config.transferEnable;
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
        if (config.backends.empty()) { return Status::InvalidParam("invalid backends"); }
        if (!config.transferEnable) { return Status::OK(); }
        if (config.ioSize == 0 || config.shardSize < config.ioSize ||
            config.blockSize < config.shardSize || config.shardSize % config.ioSize != 0 ||
            config.blockSize % config.shardSize != 0) {
            return Status::InvalidParam("invalid size({},{},{})", config.ioSize, config.shardSize,
                                        config.blockSize);
        }
        if (config.transferStreamNumber == 0) {
            return Status::InvalidParam("invalid stream number({})", config.transferStreamNumber);
        }
        return Status::OK();
    }
    void ShowConfig(const Config& config)
    {
        constexpr const char* ns = "PosixStore";
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
        UC_INFO("Set {}::Backends to {}.", ns, config.backends);
        UC_INFO("Set {}::TransferEnable to {}.", ns, config.transferEnable);
        if (!config.transferEnable) { return; }
        UC_INFO("Set {}::IOSize to {}.", ns, config.ioSize);
        UC_INFO("Set {}::ShardSize to {}.", ns, config.shardSize);
        UC_INFO("Set {}::BlockSize to {}.", ns, config.blockSize);
        UC_INFO("Set {}::TransferIoDirect to {}.", ns, config.transferIoDirect);
        UC_INFO("Set {}::TransferStreamNumber to {}.", ns, config.transferStreamNumber);
        UC_INFO("Set {}::TransferTimeoutMs to {}.", ns, config.transferTimeoutMs);
    }
};

PosixStore::PosixStore::~PosixStore() = default;

Status PosixStore::PosixStore::Setup(const Config& config)
{
    try {
        impl_ = std::make_shared<PosixStoreImpl>();
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make posix store object.", e.what());
        return Status::Error(e.what());
    }
    return impl_->Setup(config);
}

Expected<std::vector<uint8_t>> PosixStore::PosixStore::Lookup(const Detail::BlockId* blocks,
                                                              size_t num)
{
    return impl_->spaceMgr.Lookup(blocks, num);
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

}  // namespace UC::PosixStore
