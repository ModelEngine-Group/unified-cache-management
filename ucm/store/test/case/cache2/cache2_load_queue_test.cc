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
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>
#include "cache2/cc/load_queue.h"
#include "detail/mock_store.h"
#include "detail/types_helper.h"
#include "metrics_api.h"
#include "thread/latch.h"
#include "type/types.h"

namespace {

using UC::Cache2::Task;
using UC::Cache2::TaskIdSet;
using UC::Cache2::TaskPtr;
using UC::Cache2::WaiterPtr;
using UC::Test::Detail::MockStore;
using UC::Test::Detail::TypesHelper;
using SlotState = UC::Cache2::CtrlLayout::SlotMeta::State;

constexpr size_t kShardSize = 4096;
constexpr size_t kWaitMs = 5000;
char gSlotBase[64 * kShardSize];
char gDeviceSlotBase[64 * kShardSize];

class FakeBuffer {
public:
    class Handle {
    public:
        Handle(FakeBuffer* buf, size_t slotIdx, bool owner, bool hostAccessible)
            : buf_{buf}, slotIdx_{slotIdx}, owner_{owner}, hostAccessible_{hostAccessible}
        {
            buf_->liveHandles_.fetch_add(1, std::memory_order_relaxed);
        }
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        Handle(Handle&& o) noexcept
            : buf_{o.buf_},
              slotIdx_{o.slotIdx_},
              owner_{o.owner_},
              hostAccessible_{o.hostAccessible_}
        {
            o.buf_ = nullptr;
            o.slotIdx_ = 0;
            o.owner_ = false;
        }
        Handle& operator=(Handle&& o) noexcept
        {
            Handle tmp(std::move(o));
            std::swap(buf_, tmp.buf_);
            std::swap(slotIdx_, tmp.slotIdx_);
            std::swap(owner_, tmp.owner_);
            std::swap(hostAccessible_, tmp.hostAccessible_);
            return *this;
        }
        ~Handle()
        {
            if (buf_ != nullptr) { buf_->liveHandles_.fetch_sub(1, std::memory_order_relaxed); }
        }
        bool Owner() const { return owner_; }
        bool HostAccessible() const { return hostAccessible_; }
        void* Data() { return gSlotBase + slotIdx_ * kShardSize; }
        void* DeviceData() { return gDeviceSlotBase + slotIdx_ * kShardSize; }
        SlotState GetState() const { return buf_->SlotStateOf(slotIdx_); }
        void MarkReady()
        {
            if (!owner_) { return; }
            buf_->SetSlotStateByIdx(slotIdx_, SlotState::Ready);
            buf_->markedReady_.fetch_add(1, std::memory_order_relaxed);
        }
        void MarkFailed()
        {
            if (!owner_) { return; }
            buf_->SetSlotStateByIdx(slotIdx_, SlotState::Failed);
            buf_->markedFailed_.fetch_add(1, std::memory_order_relaxed);
        }

    private:
        FakeBuffer* buf_{nullptr};
        size_t slotIdx_{0};
        bool owner_{false};
        bool hostAccessible_{true};
    };
    struct GetCall {
        UC::Detail::BlockId block;
        size_t offset;
        bool allowReserved;
    };
    struct PreallocCall {
        UC::Detail::BlockId block;
        size_t offset;
        bool allowReserved;
    };

    FakeBuffer()
    {
        for (auto& state : slotStates_) {
            state.store(SlotState::Loading, std::memory_order_relaxed);
        }
    }
    Handle Get(const UC::Detail::BlockId& blockId, size_t offset, bool allowReserved = false)
    {
        getCalls_.push_back(GetCall{blockId, offset, allowReserved});
        const auto key = std::make_pair(blockId, offset);
        auto it = slots_.find(key);
        if (it != slots_.end()) { return Handle{this, it->second, false, HostAccessibleOf(key)}; }
        const auto slotIdx = nextSlotIdx_++;
        slots_.emplace(key, slotIdx);
        return Handle{this, slotIdx, true, HostAccessibleOf(key)};
    }
    void Prealloc(const UC::Detail::BlockId& blockId, size_t offset, bool allowReserved = false)
    {
        preallocCalls_.push_back(PreallocCall{blockId, offset, allowReserved});
    }
    void SetHostAccessible(bool accessible) { defaultHostAccessible_ = accessible; }
    void SetHostAccessible(const UC::Detail::BlockId& block, bool accessible, size_t offset = 0)
    {
        hostAccessibleOverrides_[std::make_pair(block, offset)] = accessible;
    }
    size_t SetExisting(const UC::Detail::BlockId& block, SlotState state = SlotState::Ready,
                       size_t offset = 0)
    {
        const auto slotIdx = nextSlotIdx_++;
        slots_.emplace(std::make_pair(block, offset), slotIdx);
        slotStates_[slotIdx].store(state, std::memory_order_relaxed);
        return slotIdx;
    }
    void SetSlotStateByIdx(size_t slotIdx, SlotState state)
    {
        slotStates_[slotIdx].store(state, std::memory_order_release);
    }
    size_t LiveHandles() const { return liveHandles_.load(std::memory_order_relaxed); }
    size_t MarkedReady() const { return markedReady_.load(std::memory_order_relaxed); }
    size_t MarkedFailed() const { return markedFailed_.load(std::memory_order_relaxed); }
    const std::vector<GetCall>& GetCalls() const { return getCalls_; }
    const std::vector<PreallocCall>& PreallocCalls() const { return preallocCalls_; }
    std::function<void(size_t)> onGetState;

private:
    SlotState SlotStateOf(size_t slotIdx)
    {
        if (onGetState) { onGetState(slotIdx); }
        return slotStates_[slotIdx].load(std::memory_order_acquire);
    }
    bool HostAccessibleOf(const std::pair<UC::Detail::BlockId, size_t>& key) const
    {
        auto it = hostAccessibleOverrides_.find(key);
        return it != hostAccessibleOverrides_.end() ? it->second : defaultHostAccessible_;
    }
    std::map<std::pair<UC::Detail::BlockId, size_t>, size_t> slots_{};
    std::map<std::pair<UC::Detail::BlockId, size_t>, bool> hostAccessibleOverrides_{};
    bool defaultHostAccessible_{true};
    std::array<std::atomic<SlotState>, 64> slotStates_{};
    size_t nextSlotIdx_{0};
    std::atomic<size_t> liveHandles_{0};
    std::atomic<size_t> markedReady_{0};
    std::atomic<size_t> markedFailed_{0};
    std::vector<GetCall> getCalls_{};
    std::vector<PreallocCall> preallocCalls_{};
};

class FakeStream {
public:
    struct ScatterCall {
        void* src{nullptr};
        std::vector<void*> dsts;
        std::vector<size_t> sizes;
    };
    static inline std::vector<ScatterCall> scatters;
    static inline std::vector<ScatterCall> d2dScatters;
    static inline std::atomic<size_t> syncs{0};
    static inline std::atomic<int32_t> lastDeviceId{-1};
    static inline std::atomic<size_t> lastStreamNumber{0};
    static inline std::atomic<size_t> setupCalls{0};
    static inline std::function<UC::Status()> onSetup;
    static inline std::function<UC::Status(const ScatterCall&)> onScatter;
    static inline std::function<UC::Status()> onSync;

    UC::Status Setup(const int32_t deviceId, const size_t streamNumber)
    {
        setupCalls.fetch_add(1, std::memory_order_relaxed);
        lastDeviceId.store(deviceId, std::memory_order_relaxed);
        lastStreamNumber.store(streamNumber, std::memory_order_relaxed);
        return onSetup ? onSetup() : UC::Status::OK();
    }
    UC::Status HostToDeviceScatterAsync(void* src, void* dst[], const std::vector<size_t>& sizes)
    {
        scatters.push_back(ScatterCall{src, std::vector<void*>(dst, dst + sizes.size()), sizes});
        return onScatter ? onScatter(scatters.back()) : UC::Status::OK();
    }
    UC::Status DeviceToDeviceScatterAsync(void* src, void* dst[], const std::vector<size_t>& sizes)
    {
        d2dScatters.push_back(ScatterCall{src, std::vector<void*>(dst, dst + sizes.size()), sizes});
        return UC::Status::OK();
    }
    UC::Status Synchronize()
    {
        syncs.fetch_add(1, std::memory_order_relaxed);
        return onSync ? onSync() : UC::Status::OK();
    }
    static void Reset()
    {
        scatters.clear();
        d2dScatters.clear();
        syncs.store(0, std::memory_order_relaxed);
        lastDeviceId.store(-1, std::memory_order_relaxed);
        lastStreamNumber.store(0, std::memory_order_relaxed);
        setupCalls.store(0, std::memory_order_relaxed);
        onSetup = nullptr;
        onScatter = nullptr;
        onSync = nullptr;
    }
};

using TestedQueue = UC::Cache2::LoadQ<FakeBuffer, FakeStream>;

UC::Detail::TaskHandle NextBackendHandle()
{
    static std::atomic<UC::Detail::TaskHandle> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

bool AwaitTrue(const std::function<bool()>& predicate, size_t timeoutMs = kWaitMs)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return predicate();
}

std::uint64_t HistogramCount(const UC::Metrics::HistogramStat& histogram)
{
    return std::accumulate(histogram.bucketCounts.begin(), histogram.bucketCounts.end(),
                           std::uint64_t{0});
}

class UCCache2LoadQueueTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        UC::Metrics::SetUp();
        UC::Metrics::CreateStats("cache_load_queue_wait_duration_ms", "histogram",
                                 {0.1, 1, 10, 100, 5000});
        UC::Metrics::CreateStats("cache_load_backend_submit_duration_ms", "histogram",
                                 {0.1, 1, 10, 100, 5000});
        UC::Metrics::CreateStats("cache_shard_backend_wait_ms", "histogram",
                                 {0.1, 1, 10, 100, 5000});
        UC::Metrics::CreateStats("cache_h2d_submit_ms", "histogram", {0.1, 1, 10, 100, 5000});
        UC::Metrics::CreateStats("cache_h2d_sync_ms", "histogram", {0.1, 1, 10, 100, 5000});
        UC::Metrics::CreateStats("cache_load_backend_shards_total", "counter");
        UC::Metrics::CreateStats("cache_load_shards_total", "counter");
        UC::Metrics::CreateStats("cache_load_wait_shards_total", "counter");
        UC::Metrics::CreateStats("cache_load_success_shards_total", "counter");
        UC::Metrics::CreateStats("cache_posix_load_success_shards_total", "counter");
        UC::Metrics::CreateStats("cache_load_failed_shards_total", "counter");
        UC::Metrics::CreateStats("cache_load_queue_full_total", "counter");
        UC::Metrics::CreateStats("cache_backend_load_submit_errors_total", "counter");
        UC::Metrics::CreateStats("cache_backend_load_wait_errors_total", "counter");
        UC::Metrics::CreateStats("cache_h2d_errors_total", "counter");
    }
    void SetUp() override
    {
        FakeStream::Reset();
        backendWaits_.store(0, std::memory_order_relaxed);
        UC::Metrics::GetAllStatsAndClear();
        config_.storeBackend = &backend_;
        config_.deviceId = 0;
        config_.tensorSizes = {kShardSize};
        config_.shardSize = kShardSize;
        config_.blockSize = kShardSize;
        config_.waitingQueueDepth = 8;
        config_.runningQueueDepth = 16;
        config_.streamNumber = 2;
        config_.localRankSize = 1;
        ASSERT_TRUE(loadQ_.Setup(config_, &failureSet_, &buffer_).Success());
    }
    static UC::Detail::TaskDesc MakeDesc(const UC::Detail::BlockId& block, size_t index = 0,
                                         uintptr_t addr = 0x1000)
    {
        UC::Detail::TaskDesc desc;
        desc.push_back({block, index, {reinterpret_cast<void*>(addr)}});
        return desc;
    }
    std::pair<TaskPtr, WaiterPtr> SubmitOne(UC::Detail::TaskDesc desc)
    {
        auto task = std::make_shared<Task>(Task::Type::LOAD, std::move(desc));
        auto waiter = std::make_shared<UC::Latch>();
        loadQ_.Submit(task, waiter);
        return {std::move(task), std::move(waiter)};
    }

    UC::Cache2::Config config_;
    TaskIdSet failureSet_;
    MockStore backend_;
    FakeBuffer buffer_;
    std::atomic<size_t> backendWaits_{0};
    TestedQueue loadQ_;
};

TEST_F(UCCache2LoadQueueTest, SetupPassesStreamParams)
{
    EXPECT_EQ(FakeStream::setupCalls.load(), 1);
    EXPECT_EQ(FakeStream::lastDeviceId.load(), 0);
    EXPECT_EQ(FakeStream::lastStreamNumber.load(), 2);
}

TEST_F(UCCache2LoadQueueTest, LoadCacheHitSkipsBackend)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    const auto slotIdx = buffer_.SetExisting(block, SlotState::Ready);
    EXPECT_CALL(backend_, Load).Times(0);
    EXPECT_CALL(backend_, Wait).Times(0);
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    ASSERT_EQ(FakeStream::scatters.size(), 1);
    EXPECT_EQ(FakeStream::scatters[0].src, gSlotBase + slotIdx * kShardSize);
    EXPECT_EQ(FakeStream::scatters[0].dsts, std::vector<void*>{reinterpret_cast<void*>(0x1000)});
    EXPECT_EQ(FakeStream::scatters[0].sizes, std::vector<size_t>{kShardSize});
    EXPECT_EQ(FakeStream::syncs.load(), 1);
    EXPECT_EQ(buffer_.MarkedReady(), 0);
    EXPECT_EQ(buffer_.MarkedFailed(), 0);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    EXPECT_EQ(counters.at("cache_load_success_shards_total"), 1);
    EXPECT_EQ(counters.at("cache_posix_load_success_shards_total"), 0);
    EXPECT_EQ(counters.at("cache_load_shards_total"), 1);
    EXPECT_EQ(counters.at("cache_load_wait_shards_total"), 0);
    EXPECT_TRUE(AwaitTrue([this] { return buffer_.LiveHandles() == 0; }));
}

TEST_F(UCCache2LoadQueueTest, LoadMissSubmitsBackendAndH2d)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    UC::Detail::TaskDesc loaded;
    EXPECT_CALL(backend_, Load).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc desc) {
        loaded = std::move(desc);
        return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
    }));
    EXPECT_CALL(backend_, Wait).WillOnce(testing::Invoke([this](UC::Detail::TaskHandle) {
        backendWaits_.fetch_add(1, std::memory_order_relaxed);
        return UC::Status::OK();
    }));
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    ASSERT_EQ(buffer_.GetCalls().size(), 1);
    EXPECT_TRUE(buffer_.GetCalls()[0].allowReserved);
    ASSERT_EQ(loaded.size(), 1);
    EXPECT_EQ(loaded.brief, "Backend2Cache");
    EXPECT_EQ(loaded[0].owner, block);
    EXPECT_EQ(loaded[0].addrs, std::vector<void*>{&gSlotBase[0]});
    ASSERT_TRUE(AwaitTrue([this] { return backendWaits_.load() == 1; }));
    EXPECT_EQ(buffer_.MarkedReady(), 1);
    EXPECT_EQ(buffer_.MarkedFailed(), 0);
    ASSERT_EQ(FakeStream::scatters.size(), 1);
    EXPECT_EQ(FakeStream::scatters[0].src, &gSlotBase[0]);
    EXPECT_EQ(FakeStream::scatters[0].dsts, std::vector<void*>{reinterpret_cast<void*>(0x1000)});
    EXPECT_EQ(FakeStream::syncs.load(), 1);
    EXPECT_TRUE(AwaitTrue([this] { return buffer_.LiveHandles() == 0; }));
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    EXPECT_EQ(counters.at("cache_load_backend_shards_total"), 1);
    EXPECT_EQ(counters.at("cache_load_wait_shards_total"), 1);
    EXPECT_EQ(counters.at("cache_posix_load_success_shards_total"), 1);
    EXPECT_EQ(counters.at("cache_load_success_shards_total"), 0);
}

TEST_F(UCCache2LoadQueueTest, LoadNonOwnerWaitsForSlotReady)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    const auto slotIdx = buffer_.SetExisting(block, SlotState::Loading);
    std::atomic<size_t> stateCalls{0};
    buffer_.onGetState = [&](size_t) { stateCalls.fetch_add(1, std::memory_order_relaxed); };
    EXPECT_CALL(backend_, Load).Times(0);
    EXPECT_CALL(backend_, Wait).Times(0);
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(AwaitTrue([&] { return stateCalls.load() >= 2; }));
    buffer_.SetSlotStateByIdx(slotIdx, SlotState::Ready);
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    buffer_.onGetState = nullptr;
    EXPECT_FALSE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::scatters.size(), 1);
    EXPECT_EQ(FakeStream::scatters[0].src, gSlotBase + slotIdx * kShardSize);
    EXPECT_EQ(buffer_.MarkedReady(), 0);
    EXPECT_EQ(buffer_.MarkedFailed(), 0);
}

TEST_F(UCCache2LoadQueueTest, LoadHostInaccessibleHitUsesD2d)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    const auto slotIdx = buffer_.SetExisting(block, SlotState::Ready);
    buffer_.SetHostAccessible(block, false);
    EXPECT_CALL(backend_, Load).Times(0);
    EXPECT_CALL(backend_, Wait).Times(0);
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::scatters.size(), 0);
    ASSERT_EQ(FakeStream::d2dScatters.size(), 1);
    EXPECT_EQ(FakeStream::d2dScatters[0].src, gDeviceSlotBase + slotIdx * kShardSize);
    EXPECT_EQ(FakeStream::d2dScatters[0].dsts, std::vector<void*>{reinterpret_cast<void*>(0x1000)});
    EXPECT_EQ(FakeStream::d2dScatters[0].sizes, std::vector<size_t>{kShardSize});
    EXPECT_EQ(FakeStream::syncs.load(), 1);
    EXPECT_EQ(buffer_.MarkedReady(), 0);
    EXPECT_EQ(buffer_.MarkedFailed(), 0);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).at("cache_load_success_shards_total"), 1);
}

TEST_F(UCCache2LoadQueueTest, LoadHostInaccessibleMissFails)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    buffer_.SetHostAccessible(block, false);
    EXPECT_CALL(backend_, Load).Times(0);
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_TRUE(failureSet_.Contains(task->id));
    EXPECT_EQ(buffer_.MarkedFailed(), 1);
    EXPECT_EQ(FakeStream::scatters.size(), 0);
    EXPECT_EQ(FakeStream::d2dScatters.size(), 0);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).at("cache_load_failed_shards_total"), 1);
}

TEST_F(UCCache2LoadQueueTest, LoadSlotFailedFailsTask)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    buffer_.SetExisting(block, SlotState::Failed);
    EXPECT_CALL(backend_, Load).Times(0);
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_TRUE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::scatters.size(), 0);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).at("cache_load_failed_shards_total"), 1);
}

TEST_F(UCCache2LoadQueueTest, LoadFailsWhenBackendSubmitFails)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    EXPECT_CALL(backend_, Load).WillOnce(testing::Invoke([] {
        return UC::Expected<UC::Detail::TaskHandle>(UC::Status::Error("backend failure"));
    }));
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_TRUE(failureSet_.Contains(task->id));
    EXPECT_EQ(buffer_.MarkedFailed(), 1);
    EXPECT_EQ(FakeStream::scatters.size(), 0);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    EXPECT_EQ(counters.at("cache_backend_load_submit_errors_total"), 1);
    EXPECT_EQ(counters.at("cache_load_failed_shards_total"), 1);
}

TEST_F(UCCache2LoadQueueTest, LoadFailsWhenBackendWaitFails)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    EXPECT_CALL(backend_, Load).WillOnce(testing::Invoke([](UC::Detail::TaskDesc) {
        return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
    }));
    EXPECT_CALL(backend_, Wait).WillOnce(testing::Invoke([this](UC::Detail::TaskHandle) {
        backendWaits_.fetch_add(1, std::memory_order_relaxed);
        return UC::Status::Error("backend wait failure");
    }));
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_TRUE(failureSet_.Contains(task->id));
    EXPECT_EQ(buffer_.MarkedFailed(), 1);
    EXPECT_EQ(FakeStream::scatters.size(), 0);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    EXPECT_EQ(counters.at("cache_backend_load_wait_errors_total"), 1);
    EXPECT_EQ(counters.at("cache_load_failed_shards_total"), 1);
}

TEST_F(UCCache2LoadQueueTest, LoadFailsWhenScatterFails)
{
    FakeStream::onScatter = [](const FakeStream::ScatterCall&) {
        return UC::Status::Error("h2d failure");
    };
    const auto block = TypesHelper::MakeBlockIdRandomly();
    EXPECT_CALL(backend_, Load).WillOnce(testing::Invoke([](UC::Detail::TaskDesc) {
        return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
    }));
    EXPECT_CALL(backend_, Wait).WillOnce(testing::Invoke([this](UC::Detail::TaskHandle) {
        backendWaits_.fetch_add(1, std::memory_order_relaxed);
        return UC::Status::OK();
    }));
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_TRUE(failureSet_.Contains(task->id));
    EXPECT_EQ(buffer_.MarkedReady(), 1);
    EXPECT_EQ(FakeStream::scatters.size(), 1);
    EXPECT_EQ(FakeStream::syncs.load(), 0);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    EXPECT_EQ(counters.at("cache_h2d_errors_total"), 1);
    EXPECT_EQ(counters.at("cache_load_failed_shards_total"), 1);
    EXPECT_TRUE(AwaitTrue([this] { return buffer_.LiveHandles() == 0; }));
}

TEST_F(UCCache2LoadQueueTest, LoadFailsWhenSyncFails)
{
    FakeStream::onSync = [] { return UC::Status::Error("sync failure"); };
    const auto block = TypesHelper::MakeBlockIdRandomly();
    EXPECT_CALL(backend_, Load).WillOnce(testing::Invoke([](UC::Detail::TaskDesc) {
        return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
    }));
    EXPECT_CALL(backend_, Wait).WillOnce(testing::Invoke([this](UC::Detail::TaskHandle) {
        backendWaits_.fetch_add(1, std::memory_order_relaxed);
        return UC::Status::OK();
    }));
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_TRUE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::scatters.size(), 1);
    EXPECT_EQ(FakeStream::syncs.load(), 1);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    EXPECT_EQ(counters.at("cache_h2d_errors_total"), 1);
    EXPECT_EQ(counters.at("cache_load_failed_shards_total"), 1);
    EXPECT_TRUE(AwaitTrue([this] { return buffer_.LiveHandles() == 0; }));
}

TEST_F(UCCache2LoadQueueTest, MultiShardsSingleSync)
{
    const auto blockA = TypesHelper::MakeBlockIdRandomly();
    const auto blockB = TypesHelper::MakeBlockIdRandomly();
    const auto blockC = TypesHelper::MakeBlockIdRandomly();
    buffer_.SetExisting(blockA, SlotState::Ready);
    buffer_.SetExisting(blockB, SlotState::Ready);
    buffer_.SetExisting(blockC, SlotState::Ready);
    EXPECT_CALL(backend_, Load).Times(0);
    EXPECT_CALL(backend_, Wait).Times(0);
    UC::Detail::TaskDesc desc;
    desc.push_back({blockA, 0, {reinterpret_cast<void*>(0x1000)}});
    desc.push_back({blockB, 0, {reinterpret_cast<void*>(0x2000)}});
    desc.push_back({blockC, 0, {reinterpret_cast<void*>(0x3000)}});
    auto [task, waiter] = SubmitOne(std::move(desc));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    ASSERT_EQ(FakeStream::scatters.size(), 3);
    EXPECT_EQ(FakeStream::scatters[0].dsts, std::vector<void*>{reinterpret_cast<void*>(0x1000)});
    EXPECT_EQ(FakeStream::scatters[1].dsts, std::vector<void*>{reinterpret_cast<void*>(0x2000)});
    EXPECT_EQ(FakeStream::scatters[2].dsts, std::vector<void*>{reinterpret_cast<void*>(0x3000)});
    EXPECT_EQ(FakeStream::syncs.load(), 1);
    EXPECT_TRUE(AwaitTrue([this] { return buffer_.LiveHandles() == 0; }));
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    EXPECT_EQ(counters.at("cache_load_success_shards_total"), 3);
    EXPECT_EQ(counters.at("cache_load_shards_total"), 3);
    EXPECT_EQ(counters.at("cache_load_wait_shards_total"), 0);
}

TEST_F(UCCache2LoadQueueTest, RearrangeOrderFollowsLocalRankInterleave)
{
    TestedQueue loadQ;
    auto config = config_;
    config.localRankSize = 4;
    config.deviceId = 2;
    ASSERT_TRUE(loadQ.Setup(config, &failureSet_, &buffer_).Success());

    const auto block = TypesHelper::MakeBlockIdRandomly();
    UC::Detail::TaskDesc desc;
    std::vector<void*> addrs(8);
    for (size_t i = 0; i < 8; ++i) {
        addrs[i] = reinterpret_cast<void*>(0x1000 * (i + 1));
        buffer_.SetExisting(block, SlotState::Ready, i);
        desc.push_back({block, i, {addrs[i]}});
    }
    EXPECT_CALL(backend_, Load).Times(0);
    EXPECT_CALL(backend_, Wait).Times(0);
    auto task = std::make_shared<Task>(Task::Type::LOAD, std::move(desc));
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    ASSERT_EQ(FakeStream::scatters.size(), 8);
    const std::vector<size_t> expectedOrder{2, 6, 3, 7, 0, 4, 1, 5};
    for (size_t i = 0; i < expectedOrder.size(); ++i) {
        EXPECT_EQ(FakeStream::scatters[i].dsts, std::vector<void*>{addrs[expectedOrder[i]]})
            << "scatter " << i;
    }
    EXPECT_EQ(FakeStream::syncs.load(), 1);
}

TEST_F(UCCache2LoadQueueTest, PreallocNextShardSlots)
{
    TestedQueue loadQ;
    auto config = config_;
    config.blockSize = 4 * kShardSize;
    ASSERT_TRUE(loadQ.Setup(config, &failureSet_, &buffer_).Success());

    const auto block = TypesHelper::MakeBlockIdRandomly();
    UC::Detail::TaskDesc desc;
    for (size_t i = 0; i < 4; ++i) {
        buffer_.SetExisting(block, SlotState::Ready, i);
        desc.push_back({block, i, {reinterpret_cast<void*>(0x1000 * (i + 1))}});
    }
    EXPECT_CALL(backend_, Load).Times(0);
    EXPECT_CALL(backend_, Wait).Times(0);
    auto task = std::make_shared<Task>(Task::Type::LOAD, std::move(desc));
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    loadQ.Close();
    const auto& preallocs = buffer_.PreallocCalls();
    ASSERT_EQ(preallocs.size(), 3);
    EXPECT_EQ(preallocs[0].block, block);
    EXPECT_EQ(preallocs[0].offset, 1);
    EXPECT_TRUE(preallocs[0].allowReserved);
    EXPECT_EQ(preallocs[1].block, block);
    EXPECT_EQ(preallocs[1].offset, 2);
    EXPECT_TRUE(preallocs[1].allowReserved);
    EXPECT_EQ(preallocs[2].block, block);
    EXPECT_EQ(preallocs[2].offset, 3);
    EXPECT_TRUE(preallocs[2].allowReserved);
}

TEST_F(UCCache2LoadQueueTest, SubmitFailsWhenWaitingQueueFull)
{
    TestedQueue loadQ;
    auto config = config_;
    config.waitingQueueDepth = 4;
    ASSERT_TRUE(loadQ.Setup(config, &failureSet_, &buffer_).Success());

    std::atomic<bool> loadEntered{false};
    UC::Latch release;
    release.Up();
    EXPECT_CALL(backend_, Load)
        .WillOnce(testing::Invoke([&](UC::Detail::TaskDesc) {
            loadEntered.store(true);
            release.WaitFor(kWaitMs);
            return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
        }))
        .WillRepeatedly(testing::Invoke([](UC::Detail::TaskDesc) {
            return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
        }));
    EXPECT_CALL(backend_, Wait).WillRepeatedly(testing::Invoke([this](UC::Detail::TaskHandle) {
        backendWaits_.fetch_add(1, std::memory_order_relaxed);
        return UC::Status::OK();
    }));

    std::vector<std::pair<TaskPtr, WaiterPtr>> tasks;
    for (size_t i = 0; i < 5; ++i) {
        auto task =
            std::make_shared<Task>(Task::Type::LOAD, MakeDesc(TypesHelper::MakeBlockIdRandomly()));
        auto waiter = std::make_shared<UC::Latch>();
        loadQ.Submit(task, waiter);
        tasks.emplace_back(std::move(task), std::move(waiter));
        if (i == 0) {
            ASSERT_TRUE(AwaitTrue([&] { return loadEntered.load(); }));
        }
    }
    const auto& rejected = tasks.back();
    ASSERT_TRUE(rejected.second->WaitFor(kWaitMs));
    EXPECT_TRUE(failureSet_.Contains(rejected.first->id));
    EXPECT_FALSE(failureSet_.Contains(tasks.front().first->id));

    release.Done();
    for (auto& [task, waiter] : tasks) {
        if (task == rejected.first) { continue; }
        ASSERT_TRUE(waiter->WaitFor(kWaitMs));
        EXPECT_FALSE(failureSet_.Contains(task->id));
    }
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).at("cache_load_queue_full_total"), 1);
}

TEST_F(UCCache2LoadQueueTest, CloseFailsPendingTasksWhenStreamSetupFailed)
{
    FakeStream::onSetup = [] { return UC::Status::Error("stream failure"); };
    TestedQueue loadQ;
    auto config = config_;
    ASSERT_TRUE(loadQ.Setup(config, &failureSet_, &buffer_).Failure());
    EXPECT_CALL(backend_, Load).WillRepeatedly(testing::Invoke([](UC::Detail::TaskDesc) {
        return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
    }));

    std::vector<std::pair<TaskPtr, WaiterPtr>> tasks;
    for (size_t i = 0; i < 2; ++i) {
        auto task =
            std::make_shared<Task>(Task::Type::LOAD, MakeDesc(TypesHelper::MakeBlockIdRandomly()));
        auto waiter = std::make_shared<UC::Latch>();
        loadQ.Submit(task, waiter);
        tasks.emplace_back(std::move(task), std::move(waiter));
    }
    loadQ.Close();
    EXPECT_EQ(FakeStream::scatters.size(), 0);
    for (auto& [task, waiter] : tasks) {
        EXPECT_TRUE(waiter->WaitFor(kWaitMs));
        EXPECT_TRUE(failureSet_.Contains(task->id));
    }
}

TEST_F(UCCache2LoadQueueTest, LoadWithoutBackendCacheHit)
{
    TestedQueue loadQ;
    auto config = config_;
    config.storeBackend = nullptr;
    ASSERT_TRUE(loadQ.Setup(config, &failureSet_, &buffer_).Success());
    EXPECT_CALL(backend_, Load).Times(0);
    EXPECT_CALL(backend_, Wait).Times(0);

    const auto block = TypesHelper::MakeBlockIdRandomly();
    const auto slotIdx = buffer_.SetExisting(block, SlotState::Ready);
    auto task = std::make_shared<Task>(Task::Type::LOAD, MakeDesc(block));
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    ASSERT_EQ(FakeStream::scatters.size(), 1);
    EXPECT_EQ(FakeStream::scatters[0].src, gSlotBase + slotIdx * kShardSize);
    EXPECT_EQ(FakeStream::syncs.load(), 1);
}

TEST_F(UCCache2LoadQueueTest, LoadWithoutBackendMissFails)
{
    TestedQueue loadQ;
    auto config = config_;
    config.storeBackend = nullptr;
    ASSERT_TRUE(loadQ.Setup(config, &failureSet_, &buffer_).Success());

    const auto block = TypesHelper::MakeBlockIdRandomly();
    auto task = std::make_shared<Task>(Task::Type::LOAD, MakeDesc(block));
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_TRUE(failureSet_.Contains(task->id));
    EXPECT_EQ(buffer_.MarkedFailed(), 1);
    EXPECT_EQ(FakeStream::scatters.size(), 0);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).at("cache_load_failed_shards_total"), 1);
}

TEST_F(UCCache2LoadQueueTest, StageMetrics)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    EXPECT_CALL(backend_, Load).WillOnce(testing::Invoke([](UC::Detail::TaskDesc) {
        return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
    }));
    EXPECT_CALL(backend_, Wait).WillOnce(testing::Invoke([this](UC::Detail::TaskHandle) {
        backendWaits_.fetch_add(1, std::memory_order_relaxed);
        return UC::Status::OK();
    }));
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    ASSERT_TRUE(AwaitTrue([this] { return backendWaits_.load() == 1; }));
    ASSERT_TRUE(AwaitTrue([this] { return buffer_.LiveHandles() == 0; }));
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    const auto& histograms = std::get<2>(stats);
    EXPECT_EQ(counters.at("cache_load_backend_shards_total"), 1);
    EXPECT_EQ(counters.at("cache_load_shards_total"), 1);
    EXPECT_EQ(counters.at("cache_load_wait_shards_total"), 1);
    EXPECT_EQ(counters.at("cache_posix_load_success_shards_total"), 1);
    EXPECT_EQ(counters.at("cache_load_success_shards_total"), 0);
    EXPECT_EQ(counters.count("cache_load_failed_shards_total"), 0);
    EXPECT_EQ(HistogramCount(histograms.at("cache_load_queue_wait_duration_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("cache_load_backend_submit_duration_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("cache_shard_backend_wait_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("cache_h2d_submit_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("cache_h2d_sync_ms")), 1);
    EXPECT_EQ(counters.count("cache_load_queue_full_total"), 0);
    EXPECT_EQ(counters.count("cache_backend_load_submit_errors_total"), 0);
    EXPECT_EQ(counters.count("cache_backend_load_wait_errors_total"), 0);
    EXPECT_EQ(counters.count("cache_h2d_errors_total"), 0);
}

}  // namespace
