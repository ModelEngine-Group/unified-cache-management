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
#include "cache/v2/dump_queue.h"
#include "detail/mock_store.h"
#include "detail/types_helper.h"
#include "metrics_api.h"
#include "thread/latch.h"
#include "type/types.h"

namespace {

using UC::Cache2::Task;
using UC::Cache2::TaskIdSet;
using UC::Cache2::TaskPtr;
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
        Handle(FakeBuffer* buf, size_t slotIdx, bool owner, SlotState state, bool hostAccessible)
            : buf_{buf},
              slotIdx_{slotIdx},
              owner_{owner},
              state_{state},
              hostAccessible_{hostAccessible}
        {
            buf_->liveHandles_.fetch_add(1, std::memory_order_relaxed);
        }
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        Handle(Handle&& o) noexcept
            : buf_{o.buf_},
              slotIdx_{o.slotIdx_},
              owner_{o.owner_},
              state_{o.state_},
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
            std::swap(state_, tmp.state_);
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
        SlotState GetState() const { return state_; }
        void MarkReady()
        {
            if (!owner_) { return; }
            state_ = SlotState::Ready;
            buf_->markedReady_.fetch_add(1, std::memory_order_relaxed);
        }
        void MarkFailed()
        {
            if (!owner_) { return; }
            state_ = SlotState::Failed;
            buf_->markedFailed_.fetch_add(1, std::memory_order_relaxed);
        }

    private:
        FakeBuffer* buf_{nullptr};
        size_t slotIdx_{0};
        bool owner_{false};
        SlotState state_{SlotState::Loading};
        bool hostAccessible_{true};
    };
    struct GetCall {
        UC::Detail::BlockId block;
        size_t offset;
        bool allowReserved;
    };

    Handle Get(const UC::Detail::BlockId& blockId, size_t offset, bool allowReserved = false)
    {
        getCalls_.push_back(GetCall{blockId, offset, allowReserved});
        const auto key = std::make_pair(blockId, offset);
        auto it = slots_.find(key);
        if (it != slots_.end()) {
            return Handle{this, it->second, false, SlotState::Ready,
                          HostAccessibleOf(blockId, offset)};
        }
        const auto slotIdx = nextSlotIdx_++;
        slots_.emplace(key, slotIdx);
        return Handle{this, slotIdx, true, ownerState_, HostAccessibleOf(blockId, offset)};
    }
    void SetOwnerState(SlotState state) { ownerState_ = state; }
    void SetHostAccessible(bool accessible) { defaultHostAccessible_ = accessible; }
    void SetHostAccessible(const UC::Detail::BlockId& block, bool accessible, size_t offset = 0)
    {
        hostAccessibleOverrides_[std::make_pair(block, offset)] = accessible;
    }
    void SetExisting(const std::vector<UC::Detail::BlockId>& blocks, size_t offset = 0)
    {
        for (const auto& block : blocks) {
            slots_.emplace(std::make_pair(block, offset), nextSlotIdx_++);
        }
    }
    size_t LiveHandles() const { return liveHandles_.load(std::memory_order_relaxed); }
    size_t MarkedReady() const { return markedReady_.load(std::memory_order_relaxed); }
    size_t MarkedFailed() const { return markedFailed_.load(std::memory_order_relaxed); }
    const std::vector<GetCall>& GetCalls() const { return getCalls_; }

private:
    bool HostAccessibleOf(const UC::Detail::BlockId& block, size_t offset) const
    {
        auto it = hostAccessibleOverrides_.find(std::make_pair(block, offset));
        return it != hostAccessibleOverrides_.end() ? it->second : defaultHostAccessible_;
    }
    std::map<std::pair<UC::Detail::BlockId, size_t>, size_t> slots_{};
    std::map<std::pair<UC::Detail::BlockId, size_t>, bool> hostAccessibleOverrides_{};
    bool defaultHostAccessible_{true};
    SlotState ownerState_{SlotState::Loading};
    size_t nextSlotIdx_{0};
    std::atomic<size_t> liveHandles_{0};
    std::atomic<size_t> markedReady_{0};
    std::atomic<size_t> markedFailed_{0};
    std::vector<GetCall> getCalls_{};
};

class FakeStream {
public:
    struct GatherCall {
        std::vector<void*> src;
        void* dst{nullptr};
        std::vector<size_t> sizes;
    };
    static inline std::vector<GatherCall> gathers;
    static inline std::vector<GatherCall> d2dGathers;
    static inline std::vector<uintptr_t> waitedEvents;
    static inline std::atomic<size_t> syncs{0};
    static inline std::atomic<int32_t> lastDeviceId{-1};
    static inline std::atomic<size_t> lastStreamNumber{0};
    static inline std::atomic<size_t> setupCalls{0};
    static inline std::function<UC::Status()> onSetup;
    static inline std::function<UC::Status(const GatherCall&)> onGather;
    static inline std::function<UC::Status()> onSync;
    static inline std::function<UC::Status(uintptr_t)> onWaitEvent;

    UC::Status Setup(const int32_t deviceId, const size_t streamNumber)
    {
        setupCalls.fetch_add(1, std::memory_order_relaxed);
        lastDeviceId.store(deviceId, std::memory_order_relaxed);
        lastStreamNumber.store(streamNumber, std::memory_order_relaxed);
        return onSetup ? onSetup() : UC::Status::OK();
    }
    UC::Status DeviceToHostGatherAsync(void* src[], void* dst, const std::vector<size_t>& sizes)
    {
        gathers.push_back(GatherCall{std::vector<void*>(src, src + sizes.size()), dst, sizes});
        return onGather ? onGather(gathers.back()) : UC::Status::OK();
    }
    UC::Status DeviceToDeviceGatherAsync(void* src[], void* dst, const std::vector<size_t>& sizes)
    {
        d2dGathers.push_back(GatherCall{std::vector<void*>(src, src + sizes.size()), dst, sizes});
        return UC::Status::OK();
    }
    UC::Status WaitEvent(const UC::Trans::Event& event)
    {
        waitedEvents.push_back(event.NativeHandle());
        return onWaitEvent ? onWaitEvent(event.NativeHandle()) : UC::Status::OK();
    }
    UC::Status Synchronize()
    {
        syncs.fetch_add(1, std::memory_order_relaxed);
        return onSync ? onSync() : UC::Status::OK();
    }
    static void Reset()
    {
        gathers.clear();
        d2dGathers.clear();
        waitedEvents.clear();
        syncs.store(0, std::memory_order_relaxed);
        lastDeviceId.store(-1, std::memory_order_relaxed);
        lastStreamNumber.store(0, std::memory_order_relaxed);
        setupCalls.store(0, std::memory_order_relaxed);
        onSetup = nullptr;
        onGather = nullptr;
        onSync = nullptr;
        onWaitEvent = nullptr;
    }
};

using TestedQueue = UC::Cache2::DumpQ<FakeBuffer, FakeStream>;

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

class UCCache2DumpQueueTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        UC::Metrics::SetUp();
        UC::Metrics::CreateStats("cache_dump_queue_wait_duration_ms", "histogram",
                                 {0.1, 1, 10, 100, 5000});
        UC::Metrics::CreateStats("cache_dump_mkbuf_duration_ms", "histogram",
                                 {0.1, 1, 10, 100, 5000});
        UC::Metrics::CreateStats("cache_d2h_duration_ms", "histogram", {0.1, 1, 10, 100, 5000});
        UC::Metrics::CreateStats("cache_dump_backend_submit_duration_ms", "histogram",
                                 {0.1, 1, 10, 100, 5000});
        UC::Metrics::CreateStats("cache_dump_backend_wait_duration_ms", "histogram",
                                 {0.1, 1, 10, 100, 5000});
        UC::Metrics::CreateStats("cache_dump_backend_shards_total", "counter");
        UC::Metrics::CreateStats("cache_dump_queue_full_total", "counter");
        UC::Metrics::CreateStats("cache_d2h_errors_total", "counter");
        UC::Metrics::CreateStats("cache_backend_dump_submit_errors_total", "counter");
        UC::Metrics::CreateStats("cache_backend_dump_wait_errors_total", "counter");
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
        ASSERT_TRUE(dumpQ_.Setup(config_, &failureSet_, &buffer_).Success());
    }
    static UC::Detail::TaskDesc MakeDesc(const UC::Detail::BlockId& block, size_t index = 0,
                                         uintptr_t addr = 0x1000)
    {
        UC::Detail::TaskDesc desc;
        desc.push_back({block, index, {reinterpret_cast<void*>(addr)}});
        return desc;
    }
    std::pair<TaskPtr, UC::Cache2::WaiterPtr> SubmitOne(UC::Detail::TaskDesc desc)
    {
        auto task = std::make_shared<Task>(Task::Type::DUMP, std::move(desc));
        auto waiter = std::make_shared<UC::Latch>();
        dumpQ_.Submit(task, waiter);
        return {std::move(task), std::move(waiter)};
    }

    UC::Cache2::Config config_;
    TaskIdSet failureSet_;
    MockStore backend_;
    FakeBuffer buffer_;
    std::atomic<size_t> backendWaits_{0};
    TestedQueue dumpQ_;
};

TEST_F(UCCache2DumpQueueTest, SetupPassesStreamParams)
{
    EXPECT_EQ(FakeStream::setupCalls.load(), 1);
    EXPECT_EQ(FakeStream::lastDeviceId.load(), 0);
    EXPECT_EQ(FakeStream::lastStreamNumber.load(), 2);
}

TEST_F(UCCache2DumpQueueTest, DumpSubmitsD2hAndBackend)
{
    const auto block = TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    UC::Detail::TaskDesc dumped;
    EXPECT_CALL(backend_, Dump).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc desc) {
        dumped = std::move(desc);
        return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
    }));
    EXPECT_CALL(backend_, Wait).WillOnce(testing::Invoke([this](UC::Detail::TaskHandle) {
        backendWaits_.fetch_add(1, std::memory_order_relaxed);
        return UC::Status::OK();
    }));
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    ASSERT_EQ(FakeStream::gathers.size(), 1);
    EXPECT_EQ(FakeStream::gathers[0].src, std::vector<void*>{reinterpret_cast<void*>(0x1000)});
    EXPECT_EQ(FakeStream::gathers[0].sizes, std::vector<size_t>{kShardSize});
    EXPECT_EQ(FakeStream::gathers[0].dst, &gSlotBase[0]);
    EXPECT_EQ(FakeStream::syncs.load(), 1);
    ASSERT_EQ(buffer_.GetCalls().size(), 1);
    EXPECT_EQ(buffer_.GetCalls()[0].block, block);
    EXPECT_EQ(buffer_.GetCalls()[0].offset, 0);
    EXPECT_FALSE(buffer_.GetCalls()[0].allowReserved);
    EXPECT_EQ(buffer_.MarkedReady(), 1);
    EXPECT_EQ(buffer_.MarkedFailed(), 0);
    ASSERT_EQ(dumped.size(), 1);
    EXPECT_EQ(dumped.brief, "Cache2Backend");
    EXPECT_EQ(dumped[0].owner, block);
    EXPECT_EQ(dumped[0].index, 0);
    ASSERT_EQ(dumped[0].addrs.size(), 1);
    EXPECT_EQ(dumped[0].addrs[0], &gSlotBase[0]);
    EXPECT_TRUE(AwaitTrue([this] { return backendWaits_.load() == 1; }));
    EXPECT_TRUE(AwaitTrue([this] { return buffer_.LiveHandles() == 0; }));
}

TEST_F(UCCache2DumpQueueTest, DumpSkipsNonOwnerShards)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    buffer_.SetExisting({block});
    EXPECT_CALL(backend_, Dump).Times(0);
    EXPECT_CALL(backend_, Wait).Times(0);
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::gathers.size(), 0);
    EXPECT_EQ(FakeStream::syncs.load(), 0);
    EXPECT_EQ(buffer_.MarkedReady(), 0);
    EXPECT_EQ(buffer_.MarkedFailed(), 0);
    EXPECT_EQ(buffer_.LiveHandles(), 0);
}

TEST_F(UCCache2DumpQueueTest, DumpSkipsD2hForReadySlots)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    buffer_.SetOwnerState(SlotState::Ready);
    EXPECT_CALL(backend_, Dump).WillOnce(testing::Invoke([](UC::Detail::TaskDesc) {
        return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
    }));
    EXPECT_CALL(backend_, Wait).WillOnce(testing::Invoke([this](UC::Detail::TaskHandle) {
        backendWaits_.fetch_add(1, std::memory_order_relaxed);
        return UC::Status::OK();
    }));
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::gathers.size(), 0);
    EXPECT_EQ(FakeStream::syncs.load(), 0);
    EXPECT_EQ(buffer_.MarkedReady(), 1);
    EXPECT_EQ(buffer_.MarkedFailed(), 0);
    EXPECT_TRUE(AwaitTrue([this] { return buffer_.LiveHandles() == 0; }));
}

TEST_F(UCCache2DumpQueueTest, DumpFailsWhenGatherFails)
{
    FakeStream::onGather = [](const FakeStream::GatherCall&) {
        return UC::Status::Error("d2h failure");
    };
    EXPECT_CALL(backend_, Dump).Times(0);
    const auto block = TypesHelper::MakeBlockIdRandomly();
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_TRUE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::gathers.size(), 1);
    EXPECT_EQ(FakeStream::syncs.load(), 0);
    EXPECT_EQ(buffer_.MarkedFailed(), 1);
    EXPECT_EQ(buffer_.MarkedReady(), 0);
    EXPECT_EQ(buffer_.LiveHandles(), 0);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).at("cache_d2h_errors_total"), 1);
}

TEST_F(UCCache2DumpQueueTest, DumpFailsWhenSyncFails)
{
    FakeStream::onSync = [] { return UC::Status::Error("sync failure"); };
    EXPECT_CALL(backend_, Dump).Times(0);
    const auto block = TypesHelper::MakeBlockIdRandomly();
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_TRUE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::gathers.size(), 1);
    EXPECT_EQ(FakeStream::syncs.load(), 1);
    EXPECT_EQ(buffer_.MarkedFailed(), 1);
    EXPECT_EQ(buffer_.MarkedReady(), 0);
    EXPECT_EQ(buffer_.LiveHandles(), 0);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).at("cache_d2h_errors_total"), 1);
}

TEST_F(UCCache2DumpQueueTest, DumpFailsWhenBackendSubmitFails)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    EXPECT_CALL(backend_, Dump).WillOnce(testing::Invoke([] {
        return UC::Expected<UC::Detail::TaskHandle>(UC::Status::Error("backend failure"));
    }));
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_TRUE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::gathers.size(), 1);
    EXPECT_EQ(FakeStream::syncs.load(), 1);
    EXPECT_EQ(buffer_.MarkedReady(), 1);
    EXPECT_EQ(buffer_.MarkedFailed(), 0);
    EXPECT_EQ(buffer_.LiveHandles(), 0);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).at("cache_backend_dump_submit_errors_total"), 1);
}

TEST_F(UCCache2DumpQueueTest, DumpFailsWhenPrerequisiteWaitFails)
{
    FakeStream::onWaitEvent = [](uintptr_t) { return UC::Status::Error("event failure"); };
    const auto block = TypesHelper::MakeBlockIdRandomly();
    auto desc = MakeDesc(block);
    desc.prerequisiteHandle = 42;
    EXPECT_CALL(backend_, Dump).Times(0);
    auto [task, waiter] = SubmitOne(std::move(desc));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_TRUE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::waitedEvents, (std::vector<uintptr_t>{42}));
    EXPECT_EQ(FakeStream::gathers.size(), 0);
    EXPECT_TRUE(buffer_.GetCalls().empty());
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).at("cache_d2h_errors_total"), 1);
}

TEST_F(UCCache2DumpQueueTest, DumpWaitsPrerequisiteEvent)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    auto desc = MakeDesc(block);
    desc.prerequisiteHandle = 42;
    EXPECT_CALL(backend_, Dump).WillOnce(testing::Invoke([](UC::Detail::TaskDesc) {
        return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
    }));
    EXPECT_CALL(backend_, Wait).WillOnce(testing::Invoke([this](UC::Detail::TaskHandle) {
        backendWaits_.fetch_add(1, std::memory_order_relaxed);
        return UC::Status::OK();
    }));
    auto [task, waiter] = SubmitOne(std::move(desc));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::waitedEvents, (std::vector<uintptr_t>{42}));
    EXPECT_EQ(FakeStream::gathers.size(), 1);
    EXPECT_TRUE(AwaitTrue([this] { return backendWaits_.load() == 1; }));
    EXPECT_TRUE(AwaitTrue([this] { return buffer_.LiveHandles() == 0; }));
}

TEST_F(UCCache2DumpQueueTest, DumpMultipleShardsInOneTask)
{
    const auto blockA = TypesHelper::MakeBlockIdRandomly();
    const auto blockB = TypesHelper::MakeBlockIdRandomly();
    UC::Detail::TaskDesc desc;
    desc.push_back({blockA, 0, {reinterpret_cast<void*>(0x1000)}});
    desc.push_back({blockB, 1, {reinterpret_cast<void*>(0x2000)}});
    UC::Detail::TaskDesc dumped;
    EXPECT_CALL(backend_, Dump).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc d) {
        dumped = std::move(d);
        return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
    }));
    EXPECT_CALL(backend_, Wait).WillOnce(testing::Invoke([this](UC::Detail::TaskHandle) {
        backendWaits_.fetch_add(1, std::memory_order_relaxed);
        return UC::Status::OK();
    }));
    auto [task, waiter] = SubmitOne(std::move(desc));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    ASSERT_EQ(FakeStream::gathers.size(), 2);
    EXPECT_EQ(FakeStream::gathers[0].src, std::vector<void*>{reinterpret_cast<void*>(0x1000)});
    EXPECT_EQ(FakeStream::gathers[1].src, std::vector<void*>{reinterpret_cast<void*>(0x2000)});
    EXPECT_NE(FakeStream::gathers[0].dst, FakeStream::gathers[1].dst);
    EXPECT_EQ(FakeStream::syncs.load(), 1);
    EXPECT_EQ(buffer_.MarkedReady(), 2);
    ASSERT_EQ(dumped.size(), 2);
    EXPECT_EQ(dumped[0].owner, blockA);
    EXPECT_EQ(dumped[1].owner, blockB);
    ASSERT_EQ(dumped[0].addrs.size(), 1);
    ASSERT_EQ(dumped[1].addrs.size(), 1);
    EXPECT_EQ(dumped[0].addrs.front(), FakeStream::gathers[0].dst);
    EXPECT_EQ(dumped[1].addrs.front(), FakeStream::gathers[1].dst);
    EXPECT_TRUE(AwaitTrue([this] { return backendWaits_.load() == 1; }));
    EXPECT_TRUE(AwaitTrue([this] { return buffer_.LiveHandles() == 0; }));
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).at("cache_dump_backend_shards_total"), 2);
}

TEST_F(UCCache2DumpQueueTest, DumpHostInaccessibleSlotUsesD2dAndSkipsBackend)
{
    buffer_.SetHostAccessible(false);
    const auto block = TypesHelper::MakeBlockIdRandomly();
    EXPECT_CALL(backend_, Dump).Times(0);
    EXPECT_CALL(backend_, Wait).Times(0);
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::gathers.size(), 0);
    ASSERT_EQ(FakeStream::d2dGathers.size(), 1);
    EXPECT_EQ(FakeStream::d2dGathers[0].src, std::vector<void*>{reinterpret_cast<void*>(0x1000)});
    EXPECT_EQ(FakeStream::d2dGathers[0].sizes, std::vector<size_t>{kShardSize});
    EXPECT_EQ(FakeStream::d2dGathers[0].dst, &gDeviceSlotBase[0]);
    EXPECT_EQ(FakeStream::syncs.load(), 1);
    EXPECT_EQ(buffer_.MarkedReady(), 1);
    EXPECT_EQ(buffer_.MarkedFailed(), 0);
    EXPECT_TRUE(AwaitTrue([this] { return buffer_.LiveHandles() == 0; }));
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).count("cache_dump_backend_shards_total"), 0);
}

TEST_F(UCCache2DumpQueueTest, DumpMixedAccessibilitySubmitsOnlyHostShards)
{
    const auto blockA = TypesHelper::MakeBlockIdRandomly();
    const auto blockB = TypesHelper::MakeBlockIdRandomly();
    buffer_.SetHostAccessible(blockB, false);
    UC::Detail::TaskDesc desc;
    desc.push_back({blockA, 0, {reinterpret_cast<void*>(0x1000)}});
    desc.push_back({blockB, 0, {reinterpret_cast<void*>(0x2000)}});
    UC::Detail::TaskDesc dumped;
    EXPECT_CALL(backend_, Dump).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc d) {
        dumped = std::move(d);
        return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
    }));
    EXPECT_CALL(backend_, Wait).WillOnce(testing::Invoke([this](UC::Detail::TaskHandle) {
        backendWaits_.fetch_add(1, std::memory_order_relaxed);
        return UC::Status::OK();
    }));
    auto [task, waiter] = SubmitOne(std::move(desc));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::gathers.size(), 1);
    ASSERT_EQ(FakeStream::d2dGathers.size(), 1);
    EXPECT_EQ(FakeStream::d2dGathers[0].src, std::vector<void*>{reinterpret_cast<void*>(0x2000)});
    EXPECT_EQ(FakeStream::d2dGathers[0].dst, &gDeviceSlotBase[kShardSize]);
    EXPECT_EQ(FakeStream::syncs.load(), 1);
    EXPECT_EQ(buffer_.MarkedReady(), 2);
    ASSERT_EQ(dumped.size(), 1);
    EXPECT_EQ(dumped[0].owner, blockA);
    EXPECT_EQ(dumped[0].addrs, std::vector<void*>{&gSlotBase[0]});
    EXPECT_TRUE(AwaitTrue([this] { return backendWaits_.load() == 1; }));
    EXPECT_TRUE(AwaitTrue([this] { return buffer_.LiveHandles() == 0; }));
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).at("cache_dump_backend_shards_total"), 1);
}

TEST_F(UCCache2DumpQueueTest, BackendWaitFailureReleasesHandles)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    EXPECT_CALL(backend_, Dump).WillOnce(testing::Invoke([](UC::Detail::TaskDesc) {
        return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
    }));
    EXPECT_CALL(backend_, Wait).WillOnce(testing::Invoke([this](UC::Detail::TaskHandle) {
        backendWaits_.fetch_add(1, std::memory_order_relaxed);
        return UC::Status::Error("backend wait failure");
    }));
    auto [task, waiter] = SubmitOne(MakeDesc(block));
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    EXPECT_TRUE(AwaitTrue([this] { return backendWaits_.load() == 1; }));
    EXPECT_TRUE(AwaitTrue([this] { return buffer_.LiveHandles() == 0; }));
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).at("cache_backend_dump_wait_errors_total"), 1);
}

TEST_F(UCCache2DumpQueueTest, SubmitFailsWhenWaitingQueueFull)
{
    TestedQueue dumpQ;
    auto config = config_;
    config.waitingQueueDepth = 4;
    ASSERT_TRUE(dumpQ.Setup(config, &failureSet_, &buffer_).Success());

    std::atomic<bool> gatherEntered{false};
    UC::Latch release;
    release.Up();
    FakeStream::onGather = [&](const FakeStream::GatherCall&) {
        gatherEntered.store(true);
        release.WaitFor(kWaitMs);
        return UC::Status::OK();
    };
    EXPECT_CALL(backend_, Dump).WillRepeatedly(testing::Invoke([](UC::Detail::TaskDesc) {
        return UC::Expected<UC::Detail::TaskHandle>(NextBackendHandle());
    }));
    EXPECT_CALL(backend_, Wait).WillRepeatedly(testing::Invoke([this](UC::Detail::TaskHandle) {
        backendWaits_.fetch_add(1, std::memory_order_relaxed);
        return UC::Status::OK();
    }));

    const auto block = TypesHelper::MakeBlockIdRandomly();
    std::vector<std::pair<TaskPtr, UC::Cache2::WaiterPtr>> tasks;
    for (size_t i = 0; i < 5; ++i) {
        auto task = std::make_shared<Task>(Task::Type::DUMP, MakeDesc(block, i));
        auto waiter = std::make_shared<UC::Latch>();
        dumpQ.Submit(task, waiter);
        tasks.emplace_back(std::move(task), std::move(waiter));
        if (i == 0) {
            ASSERT_TRUE(AwaitTrue([&] { return gatherEntered.load(); }));
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
    EXPECT_EQ(std::get<0>(stats).at("cache_dump_queue_full_total"), 1);
}

TEST_F(UCCache2DumpQueueTest, CloseFailsPendingTasksWhenStreamSetupFailed)
{
    FakeStream::onSetup = [] { return UC::Status::Error("stream failure"); };
    TestedQueue dumpQ;
    auto config = config_;
    ASSERT_TRUE(dumpQ.Setup(config, &failureSet_, &buffer_).Failure());

    const auto block = TypesHelper::MakeBlockIdRandomly();
    std::vector<std::pair<TaskPtr, UC::Cache2::WaiterPtr>> tasks;
    for (size_t i = 0; i < 3; ++i) {
        auto task = std::make_shared<Task>(Task::Type::DUMP, MakeDesc(block, i));
        auto waiter = std::make_shared<UC::Latch>();
        dumpQ.Submit(task, waiter);
        tasks.emplace_back(std::move(task), std::move(waiter));
    }
    dumpQ.Close();
    EXPECT_EQ(FakeStream::gathers.size(), 0);
    for (auto& [task, waiter] : tasks) {
        EXPECT_TRUE(waiter->WaitFor(kWaitMs));
        EXPECT_TRUE(failureSet_.Contains(task->id));
    }
}

TEST_F(UCCache2DumpQueueTest, DumpWithoutBackendWritesCacheOnly)
{
    TestedQueue dumpQ;
    auto config = config_;
    config.storeBackend = nullptr;
    ASSERT_TRUE(dumpQ.Setup(config, &failureSet_, &buffer_).Success());
    EXPECT_CALL(backend_, Dump).Times(0);
    EXPECT_CALL(backend_, Wait).Times(0);

    const auto block = TypesHelper::MakeBlockIdRandomly();
    auto task = std::make_shared<Task>(Task::Type::DUMP, MakeDesc(block));
    auto waiter = std::make_shared<UC::Latch>();
    dumpQ.Submit(task, waiter);
    ASSERT_TRUE(waiter->WaitFor(kWaitMs));
    EXPECT_FALSE(failureSet_.Contains(task->id));
    EXPECT_EQ(FakeStream::gathers.size(), 1);
    EXPECT_EQ(FakeStream::syncs.load(), 1);
    EXPECT_EQ(buffer_.MarkedReady(), 1);
    EXPECT_EQ(buffer_.MarkedFailed(), 0);
    EXPECT_EQ(buffer_.LiveHandles(), 0);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).count("cache_dump_backend_shards_total"), 0);
}

TEST_F(UCCache2DumpQueueTest, DumpStageMetrics)
{
    const auto block = TypesHelper::MakeBlockIdRandomly();
    EXPECT_CALL(backend_, Dump).WillOnce(testing::Invoke([](UC::Detail::TaskDesc) {
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
    EXPECT_EQ(counters.at("cache_dump_backend_shards_total"), 1);
    EXPECT_EQ(HistogramCount(histograms.at("cache_dump_queue_wait_duration_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("cache_dump_mkbuf_duration_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("cache_d2h_duration_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("cache_dump_backend_submit_duration_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("cache_dump_backend_wait_duration_ms")), 1);
    EXPECT_EQ(counters.count("cache_dump_queue_full_total"), 0);
    EXPECT_EQ(counters.count("cache_d2h_errors_total"), 0);
    EXPECT_EQ(counters.count("cache_backend_dump_submit_errors_total"), 0);
    EXPECT_EQ(counters.count("cache_backend_dump_wait_errors_total"), 0);
}

}  // namespace
