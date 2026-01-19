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
#include "fake_store.h"
#include <algorithm>
#include <atomic>
#include "logger/logger.h"
#include "meta_manager.h"
#include "time/stopwatch.h"

namespace UC::FakeStore {

class FakeStoreImpl {
    MetaManager metaMgr_;

public:
    static Detail::TaskHandle NextId() noexcept
    {
        static std::atomic<Detail::TaskHandle> idSeed{1};
        return idSeed.fetch_add(1, std::memory_order_relaxed);
    };
    Status Setup(const Config& config)
    {
        auto s = CheckConfig(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed to check config params: {}.", s);
            return s;
        }
        s = metaMgr_.Setup(config);
        if (s.Failure()) [[unlikely]] { return s; }
        ShowConfig(config);
        return Status::OK();
    }
    std::string Readme() const { return "FakeStore"; }
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num)
    {
        std::vector<uint8_t> founds(num);
        StopWatch sw;
        std::transform(blocks, blocks + num, founds.begin(),
                       [this](const Detail::BlockId& block) { return metaMgr_.Exist(block); });
        UC_DEBUG("Fake lookup({}) costs {:.3f}ms.", num, sw.Elapsed().count() * 1e3);
        return founds;
    }
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num)
    {
        ssize_t index = -1;
        StopWatch sw;
        for (size_t i = 0; i < num && metaMgr_.Exist(blocks[i]); i++) {
            index = static_cast<ssize_t>(i);
        }
        UC_DEBUG("Fake Lookup({}/{}) costs {:.3f}ms.", index, num, sw.Elapsed().count() * 1e3);
        return index;
    }
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task)
    {
        StopWatch sw;
        std::for_each(task.begin(), task.end(),
                      [this](const Detail::Shard& shard) { metaMgr_.Insert(shard.owner); });
        UC_DEBUG("Fake dump({}) costs {:.3f}ms.", task.size(), sw.Elapsed().count() * 1e3);
        return NextId();
    }

private:
    Status CheckConfig(const Config& config)
    {
        if (config.uniqueId.empty()) { return Status::InvalidParam("invalid unique id"); }
        if (config.bufferNumber < 1024) {
            return Status::InvalidParam("too small buffer number({})", config.bufferNumber);
        }
        if (!config.shareBufferEnable) { return Status::InvalidParam("buffer must be shared"); }
        return Status::OK();
    }
    void ShowConfig(const Config& config)
    {
        const auto& ns = Readme();
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
        UC_INFO("Set {}::UniqueId to {}.", ns, config.uniqueId);
        UC_INFO("Set {}::BufferNumber to {}.", ns, config.bufferNumber);
        UC_INFO("Set {}::ShareBufferEnable to {}.", ns, config.shareBufferEnable);
    }
};

Status FakeStore::Setup(const Config& config)
{
    try {
        impl_ = std::make_shared<FakeStoreImpl>();
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make FakeStore object.", e.what());
        return Status::Error(e.what());
    }
    return impl_->Setup(config);
}

std::string FakeStore::Readme() const { return impl_->Readme(); }

Expected<std::vector<uint8_t>> FakeStore::Lookup(const Detail::BlockId* blocks, size_t num)
{
    return impl_->Lookup(blocks, num);
}

Expected<ssize_t> FakeStore::LookupOnPrefix(const Detail::BlockId* blocks, size_t num)
{
    return impl_->LookupOnPrefix(blocks, num);
}

Expected<Detail::TaskHandle> FakeStore::Load(Detail::TaskDesc task)
{
    return FakeStoreImpl::NextId();
}

Expected<Detail::TaskHandle> FakeStore::Dump(Detail::TaskDesc task) { return impl_->Dump(task); }

}  // namespace UC::FakeStore
