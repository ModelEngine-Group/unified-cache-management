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
 */
#include "delegator/cc/delegator_executor.h"
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace UC::Delegator {
namespace {

class ExecutorTest : public ::testing::Test {
public:
    static void SetUpTestSuite()
    {
        const auto ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
            FAIL() << "aclInit failed: " << ret;
        }
        ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    }

    static void TearDownTestSuite()
    {
        (void)aclrtResetDevice(0);
        (void)aclFinalize();
    }

    static std::shared_ptr<void> MakeDeviceBuffer(std::size_t size)
    {
        void* address = nullptr;
        if (aclrtMalloc(&address, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH) != ACL_SUCCESS) {
            return nullptr;
        }
        return std::shared_ptr<void>(address, [](void* ptr) { (void)aclrtFree(ptr); });
    }
};

class FakeTransferEndpoint final : public TransferEndpoint {
public:
    struct Call {
        Operation operation;
        std::vector<std::size_t> shards;
    };

    explicit FakeTransferEndpoint(std::size_t payload_size) : payload_size_{payload_size} {}

    Status SetupTransferRegion(const TransferRegion& region) override
    {
        if (region.device_addr == nullptr || region.size == 0) {
            return Status::InvalidParam();
        }
        return Status::OK();
    }

    void ResetTransferRegion() noexcept override {}

    Expected<Detail::TaskHandle> SubmitLoad(
        const std::vector<TransferBuffer>& buffers) override
    {
        if (buffers.empty()) { return Status::InvalidParam(); }

        std::vector<std::size_t> batch;
        std::vector<std::vector<std::uint8_t>> data;
        batch.reserve(buffers.size());
        data.reserve(buffers.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& buffer : buffers) {
                batch.push_back(buffer.index);
                load_submissions_.push_back(buffer.index);
            }
            calls_.push_back(Call{Operation::LOAD, batch});
            if (std::find(batch.begin(), batch.end(), fail_load_index_) != batch.end()) {
                return Status::Error("injected load submission failure");
            }
            for (const auto index : batch) { data.push_back(stored_[index]); }
        }

        for (std::size_t index = 0; index < buffers.size(); ++index) {
            if (data[index].size() != payload_size_) {
                return Status::InvalidParam("missing fake load data");
            }
            const auto ret = aclrtMemcpy(buffers[index].slot.device_addr, payload_size_,
                                         data[index].data(),
                                         data[index].size(), ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) { return Status::Error("fake load copy failed"); }
        }
        return NewTask(false, Status::OK(), buffers.front().index);
    }

    Expected<Detail::TaskHandle> SubmitDump(
        const std::vector<TransferBuffer>& buffers) override
    {
        if (buffers.empty()) { return Status::InvalidParam(); }

        std::vector<std::size_t> batch;
        batch.reserve(buffers.size());
        for (const auto& buffer : buffers) {
            std::vector<std::uint8_t> data(payload_size_);
            const auto ret = aclrtMemcpy(data.data(), data.size(), buffer.slot.device_addr,
                                         data.size(), ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) { return Status::Error("fake store copy failed"); }
            batch.push_back(buffer.index);

            std::lock_guard<std::mutex> lock(mutex_);
            stored_[buffer.index] = std::move(data);
            dump_slots_.push_back(buffer.slot);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            calls_.push_back(Call{Operation::DUMP, batch});
            store_batches_.push_back(std::move(batch));
        }
        const auto groupIndex = buffers.front().index;
        if (groupIndex == fail_dump_submit_index_) {
            return Status::Error("injected dump submission failure");
        }
        const auto waitStatus =
            fail_first_store_ || groupIndex == fail_dump_wait_index_
                ? Status::Error("injected store failure")
                : Status::OK();
        return NewTask(true, waitStatus, groupIndex);
    }

    Status Wait(Detail::TaskHandle task) override
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto iter = tasks_.find(task);
        if (iter == tasks_.end()) { return Status::NotFound(); }
        if (iter->second.is_store && block_first_store_ && !store_wait_seen_) {
            store_wait_seen_ = true;
            wait_entered_.notify_all();
            wait_released_.wait(lock, [this]() { return release_store_; });
        }
        return iter->second.status;
    }

    Expected<bool> Check(Detail::TaskHandle task) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto iter = tasks_.find(task);
        if (iter == tasks_.end()) { return Status::NotFound(); }

        const auto& entry = iter->second;
        if (std::find(checked_load_groups_.begin(), checked_load_groups_.end(),
                      entry.group_index) == checked_load_groups_.end()) {
            checked_load_groups_.push_back(entry.group_index);
            wait_entered_.notify_all();
        }
        if (entry.blocked && !release_load_) { return false; }
        if (entry.group_index == fail_load_check_index_) {
            return Status::Error("injected load check failure");
        }
        if (entry.status.Failure()) { return entry.status; }
        return true;
    }

    void SetData(std::size_t index, std::vector<std::uint8_t> data)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stored_[index] = std::move(data);
    }

    void BlockFirstStore()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        block_first_store_ = true;
    }

    void BlockFirstLoad()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        block_first_load_ = true;
    }

    bool WaitForStoreWait() { return WaitFor(store_wait_seen_); }

    bool WaitForLoadGroupCheck(std::size_t group)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return wait_entered_.wait_for(lock, std::chrono::seconds(2), [this, group]() {
            return std::find(checked_load_groups_.begin(), checked_load_groups_.end(), group) !=
                   checked_load_groups_.end();
        });
    }

    void ReleaseStore()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        release_store_ = true;
        wait_released_.notify_all();
    }

    void ReleaseLoad()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        release_load_ = true;
        wait_released_.notify_all();
    }

    std::vector<std::vector<std::size_t>> StoreBatches()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return store_batches_;
    }

    std::vector<std::size_t> LoadSubmissions()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return load_submissions_;
    }

    std::vector<std::size_t> CheckedLoadGroups()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return checked_load_groups_;
    }

    std::vector<Call> Calls()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return calls_;
    }

    std::vector<BufferPool::Slot> DumpSlots()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return dump_slots_;
    }

    std::size_t fail_load_index_{std::numeric_limits<std::size_t>::max()};
    std::size_t fail_load_check_index_{std::numeric_limits<std::size_t>::max()};
    std::size_t fail_dump_submit_index_{std::numeric_limits<std::size_t>::max()};
    std::size_t fail_dump_wait_index_{std::numeric_limits<std::size_t>::max()};
    bool fail_first_store_{false};

private:
    struct Task {
        bool is_store;
        Status status;
        std::size_t group_index;
        bool blocked;
    };

    Expected<Detail::TaskHandle> NewTask(
        bool is_store, Status status,
        std::size_t group_index = std::numeric_limits<std::size_t>::max())
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto task = next_task_++;
        const auto blocked = !is_store && block_first_load_ && !blocked_load_assigned_;
        if (blocked) { blocked_load_assigned_ = true; }
        tasks_.emplace(task, Task{is_store, std::move(status), group_index, blocked});
        return Detail::TaskHandle{task};
    }

    bool WaitFor(bool& ready)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return wait_entered_.wait_for(lock, std::chrono::seconds(2), [&ready]() { return ready; });
    }

    std::size_t payload_size_;
    std::mutex mutex_;
    std::condition_variable wait_entered_;
    std::condition_variable wait_released_;
    Detail::TaskHandle next_task_{1};
    std::unordered_map<Detail::TaskHandle, Task> tasks_;
    std::unordered_map<std::size_t, std::vector<std::uint8_t>> stored_;
    std::vector<std::vector<std::size_t>> store_batches_;
    std::vector<std::size_t> load_submissions_;
    std::vector<std::size_t> checked_load_groups_;
    std::vector<Call> calls_;
    std::vector<BufferPool::Slot> dump_slots_;
    bool block_first_store_{false};
    bool block_first_load_{false};
    bool blocked_load_assigned_{false};
    bool store_wait_seen_{false};
    bool release_store_{false};
    bool release_load_{false};
};

class TransferEndpointRef final : public TransferEndpoint {
public:
    explicit TransferEndpointRef(TransferEndpoint& endpoint) : endpoint_{endpoint} {}

    Status SetupTransferRegion(const TransferRegion& region) override
    {
        return endpoint_.SetupTransferRegion(region);
    }

    void ResetTransferRegion() noexcept override { endpoint_.ResetTransferRegion(); }

    Expected<Detail::TaskHandle> SubmitLoad(
        const std::vector<TransferBuffer>& buffers) override
    {
        return endpoint_.SubmitLoad(buffers);
    }

    Expected<Detail::TaskHandle> SubmitDump(
        const std::vector<TransferBuffer>& buffers) override
    {
        return endpoint_.SubmitDump(buffers);
    }

    Expected<bool> Check(Detail::TaskHandle task) override
    {
        return endpoint_.Check(task);
    }

    Status Wait(Detail::TaskHandle task) override { return endpoint_.Wait(task); }

private:
    TransferEndpoint& endpoint_;
};

Expected<std::unique_ptr<Executor>> CreateTestExecutor(
    FakeTransferEndpoint& endpoint, std::vector<std::size_t> tensor_sizes,
    std::int32_t device_id, std::size_t slot_num,
    std::size_t stream_number = Executor::kDefaultStreamNumber)
{
    return Executor::Create(std::make_unique<TransferEndpointRef>(endpoint),
                            std::move(tensor_sizes), device_id, slot_num,
                            stream_number);
}

struct DeviceShard {
    std::shared_ptr<void> first;
    std::shared_ptr<void> second;
    Detail::Shard desc;
};

DeviceShard MakeShard(std::size_t index, std::uint8_t first_value = 0)
{
    DeviceShard shard;
    shard.first = ExecutorTest::MakeDeviceBuffer(3);
    shard.second = ExecutorTest::MakeDeviceBuffer(5);
    std::array<std::uint8_t, 3> first{first_value, static_cast<std::uint8_t>(first_value + 1),
                                      static_cast<std::uint8_t>(first_value + 2)};
    std::array<std::uint8_t, 5> second{
        static_cast<std::uint8_t>(first_value + 3), static_cast<std::uint8_t>(first_value + 4),
        static_cast<std::uint8_t>(first_value + 5), static_cast<std::uint8_t>(first_value + 6),
        static_cast<std::uint8_t>(first_value + 7)};
    (void)aclrtMemcpy(shard.first.get(), first.size(), first.data(), first.size(),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(shard.second.get(), second.size(), second.data(), second.size(),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    shard.desc = Detail::Shard{
        {                  },
        index, { shard.first.get(), shard.second.get()}
    };
    return shard;
}

DeviceShard MakeEmptyShard(std::size_t index)
{
    auto shard = MakeShard(index);
    (void)aclrtMemset(shard.first.get(), 3, 0, 3);
    (void)aclrtMemset(shard.second.get(), 5, 0, 5);
    return shard;
}

Detail::TaskDesc MakeTask(std::vector<DeviceShard>& shards)
{
    Detail::TaskDesc task;
    for (auto& shard : shards) { task.push_back(shard.desc); }
    return task;
}

std::array<std::uint8_t, 8> ReadShard(const DeviceShard& shard)
{
    std::array<std::uint8_t, 8> result{};
    (void)aclrtMemcpy(result.data(), 3, shard.first.get(), 3, ACL_MEMCPY_DEVICE_TO_HOST);
    (void)aclrtMemcpy(result.data() + 3, 5, shard.second.get(), 5, ACL_MEMCPY_DEVICE_TO_HOST);
    return result;
}

bool WaitForShardData(const DeviceShard& shard, const std::array<std::uint8_t, 8>& expected)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        if (ReadShard(shard) == expected) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

TEST_F(ExecutorTest, DumpBatchUsesAvailableSlotsAndOneStoreBatch)
{
    FakeTransferEndpoint store(8);
    auto created = CreateTestExecutor(store, {3, 5}, 0, 3, 2);
    ASSERT_TRUE(created);
    auto executor = std::move(created).Value();
    std::vector<DeviceShard> shards;
    for (std::size_t index = 0; index < 5; ++index) {
        shards.push_back(MakeShard(index, static_cast<std::uint8_t>(index * 10 + 1)));
    }

    auto task = executor->Submit(MakeTask(shards), Operation::DUMP);
    ASSERT_TRUE(task);
    ASSERT_TRUE(executor->Wait(task.Value()).Success());
    EXPECT_EQ(store.StoreBatches(), (std::vector<std::vector<std::size_t>>{
                                          {0, 1, 2},
                                          {3, 4 }
    }));

    const auto slots = store.DumpSlots();
    ASSERT_EQ(slots.size(), std::size_t{5});
    std::vector<std::size_t> firstBatchOffsets;
    for (std::size_t index = 0; index < 3; ++index) {
        EXPECT_EQ(slots[index].length, std::size_t{8});
        EXPECT_EQ(slots[index].offset % (16 * 1024), std::size_t{0});
        firstBatchOffsets.push_back(slots[index].offset);
    }
    std::sort(firstBatchOffsets.begin(), firstBatchOffsets.end());
    EXPECT_EQ(firstBatchOffsets, (std::vector<std::size_t>{0, 16 * 1024, 32 * 1024}));
}

TEST_F(ExecutorTest, LoadScattersCompletedGroupsWhileFirstIsPending)
{
    FakeTransferEndpoint store(8);
    store.BlockFirstLoad();
    store.BlockFirstStore();
    store.SetData(0, {1, 2, 3, 4, 5, 6, 7, 8});
    store.SetData(1, {11, 12, 13, 14, 15, 16, 17, 18});
    store.SetData(2, {21, 22, 23, 24, 25, 26, 27, 28});
    auto created = CreateTestExecutor(store, {3, 5}, 0, 3, 2);
    ASSERT_TRUE(created);
    auto executor = std::move(created).Value();

    std::vector<DeviceShard> blockerShards{MakeShard(100), MakeShard(101), MakeShard(102)};
    auto blocker = executor->Submit(MakeTask(blockerShards), Operation::DUMP);
    ASSERT_TRUE(blocker);
    const auto storeWaitSeen = store.WaitForStoreWait();
    if (!storeWaitSeen) { store.ReleaseStore(); }
    ASSERT_TRUE(storeWaitSeen);

    std::vector<DeviceShard> firstShard{MakeEmptyShard(0)};
    std::vector<DeviceShard> secondShard{MakeEmptyShard(1)};
    std::vector<DeviceShard> thirdShard{MakeEmptyShard(2)};
    auto first = executor->Submit(MakeTask(firstShard), Operation::LOAD);
    auto second = executor->Submit(MakeTask(secondShard), Operation::LOAD);
    auto third = executor->Submit(MakeTask(thirdShard), Operation::LOAD);
    store.ReleaseStore();
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    ASSERT_TRUE(third);

    ASSERT_TRUE(executor->Wait(blocker.Value()).Success());
    const auto firstChecked = store.WaitForLoadGroupCheck(0);
    const auto thirdChecked = store.WaitForLoadGroupCheck(2);
    if (!firstChecked || !thirdChecked) { store.ReleaseLoad(); }
    ASSERT_TRUE(firstChecked);
    ASSERT_TRUE(thirdChecked);
    EXPECT_EQ(store.LoadSubmissions(), (std::vector<std::size_t>{0, 1, 2}));
    const std::array<std::uint8_t, 8> thirdData{21, 22, 23, 24, 25, 26, 27, 28};
    EXPECT_TRUE(WaitForShardData(thirdShard[0], thirdData));
    EXPECT_EQ(ReadShard(firstShard[0]), (std::array<std::uint8_t, 8>{}));
    store.ReleaseLoad();
    ASSERT_TRUE(executor->Wait(first.Value()).Success());
    ASSERT_TRUE(executor->Wait(second.Value()).Success());
    ASSERT_TRUE(executor->Wait(third.Value()).Success());
    EXPECT_EQ(ReadShard(thirdShard[0]), thirdData);
}

TEST_F(ExecutorTest, LoadSubmitFailureDoesNotScatterGroup)
{
    FakeTransferEndpoint store(8);
    store.SetData(0, {1, 2, 3, 4, 5, 6, 7, 8});
    store.SetData(1, {11, 12, 13, 14, 15, 16, 17, 18});
    store.SetData(2, {21, 22, 23, 24, 25, 26, 27, 28});
    store.fail_load_index_ = 1;
    std::vector<DeviceShard> shards{MakeEmptyShard(0), MakeEmptyShard(1), MakeEmptyShard(2)};
    auto created = CreateTestExecutor(store, {3, 5}, 0, 3, 2);
    ASSERT_TRUE(created);
    auto executor = std::move(created).Value();

    auto task = executor->Submit(MakeTask(shards), Operation::LOAD);
    ASSERT_TRUE(task);
    EXPECT_TRUE(executor->Wait(task.Value()).Failure());
    EXPECT_EQ(store.LoadSubmissions(), (std::vector<std::size_t>{0, 1, 2}));
    const auto calls = store.Calls();
    ASSERT_EQ(calls.size(), std::size_t{1});
    EXPECT_EQ(calls[0].shards, (std::vector<std::size_t>{0, 1, 2}));
    EXPECT_EQ(ReadShard(shards[0]), (std::array<std::uint8_t, 8>{}));
    EXPECT_EQ(ReadShard(shards[1]), (std::array<std::uint8_t, 8>{}));
    EXPECT_EQ(ReadShard(shards[2]), (std::array<std::uint8_t, 8>{}));
}

TEST_F(ExecutorTest, LoadCheckFailureDoesNotScatterGroup)
{
    FakeTransferEndpoint store(8);
    store.fail_load_check_index_ = 0;
    std::vector<DeviceShard> shards;
    for (std::size_t index = 0; index < 3; ++index) {
        shards.push_back(MakeEmptyShard(index));
        store.SetData(
            index,
            {static_cast<std::uint8_t>(index * 10 + 1), static_cast<std::uint8_t>(index * 10 + 2),
             static_cast<std::uint8_t>(index * 10 + 3), static_cast<std::uint8_t>(index * 10 + 4),
             static_cast<std::uint8_t>(index * 10 + 5), static_cast<std::uint8_t>(index * 10 + 6),
             static_cast<std::uint8_t>(index * 10 + 7), static_cast<std::uint8_t>(index * 10 + 8)});
    }
    auto created = CreateTestExecutor(store, {3, 5}, 0, 3, 2);
    ASSERT_TRUE(created);
    auto executor = std::move(created).Value();

    auto task = executor->Submit(MakeTask(shards), Operation::LOAD);
    ASSERT_TRUE(task);
    EXPECT_TRUE(executor->Wait(task.Value()).Failure());
    EXPECT_EQ(store.CheckedLoadGroups(), (std::vector<std::size_t>{0}));
    EXPECT_EQ(ReadShard(shards[0]), (std::array<std::uint8_t, 8>{}));
    EXPECT_EQ(ReadShard(shards[1]), (std::array<std::uint8_t, 8>{}));
    EXPECT_EQ(ReadShard(shards[2]), (std::array<std::uint8_t, 8>{}));
}

TEST_F(ExecutorTest, LoadSubmitFailureDoesNotCancelNextTransferGroup)
{
    FakeTransferEndpoint store(8);
    store.BlockFirstStore();
    store.fail_load_index_ = 0;
    store.SetData(2, {21, 22, 23, 24, 25, 26, 27, 28});

    auto created = CreateTestExecutor(store, {3, 5}, 0, 3, 2);
    ASSERT_TRUE(created);
    auto executor = std::move(created).Value();

    std::vector<DeviceShard> blockerShards{MakeShard(100), MakeShard(101), MakeShard(102)};
    auto blocker = executor->Submit(MakeTask(blockerShards), Operation::DUMP);
    ASSERT_TRUE(blocker);
    const auto storeWaitSeen = store.WaitForStoreWait();
    if (!storeWaitSeen) { store.ReleaseStore(); }
    ASSERT_TRUE(storeWaitSeen);

    std::vector<DeviceShard> failedShards{MakeEmptyShard(0), MakeEmptyShard(1)};
    std::vector<DeviceShard> successfulShards{MakeEmptyShard(2)};
    auto failed = executor->Submit(MakeTask(failedShards), Operation::LOAD);
    auto successful = executor->Submit(MakeTask(successfulShards), Operation::LOAD);
    ASSERT_TRUE(failed);
    ASSERT_TRUE(successful);

    store.ReleaseStore();
    EXPECT_TRUE(executor->Wait(blocker.Value()).Success());
    EXPECT_TRUE(executor->Wait(failed.Value()).Failure());
    EXPECT_TRUE(executor->Wait(successful.Value()).Success());
    EXPECT_EQ(store.LoadSubmissions(), (std::vector<std::size_t>{0, 1, 2}));
    EXPECT_EQ(ReadShard(successfulShards[0]),
              (std::array<std::uint8_t, 8>{21, 22, 23, 24, 25, 26, 27, 28}));
}

TEST_F(ExecutorTest, LoadCheckFailureDoesNotFailNextTransferGroup)
{
    FakeTransferEndpoint store(8);
    store.BlockFirstStore();
    store.fail_load_check_index_ = 0;
    store.SetData(0, {1, 2, 3, 4, 5, 6, 7, 8});
    store.SetData(1, {11, 12, 13, 14, 15, 16, 17, 18});

    auto created = CreateTestExecutor(store, {3, 5}, 0, 2, 2);
    ASSERT_TRUE(created);
    auto executor = std::move(created).Value();

    std::vector<DeviceShard> blockerShards{MakeShard(100), MakeShard(101)};
    auto blocker = executor->Submit(MakeTask(blockerShards), Operation::DUMP);
    ASSERT_TRUE(blocker);
    const auto storeWaitSeen = store.WaitForStoreWait();
    if (!storeWaitSeen) { store.ReleaseStore(); }
    ASSERT_TRUE(storeWaitSeen);

    std::vector<DeviceShard> failedShards{MakeEmptyShard(0)};
    std::vector<DeviceShard> successfulShards{MakeEmptyShard(1)};
    auto failed = executor->Submit(MakeTask(failedShards), Operation::LOAD);
    auto successful = executor->Submit(MakeTask(successfulShards), Operation::LOAD);
    ASSERT_TRUE(failed);
    ASSERT_TRUE(successful);

    store.ReleaseStore();
    EXPECT_TRUE(executor->Wait(blocker.Value()).Success());
    EXPECT_TRUE(executor->Wait(failed.Value()).Failure());
    EXPECT_TRUE(executor->Wait(successful.Value()).Success());
    EXPECT_EQ(store.LoadSubmissions(), (std::vector<std::size_t>{0, 1}));
    EXPECT_EQ(ReadShard(successfulShards[0]),
              (std::array<std::uint8_t, 8>{11, 12, 13, 14, 15, 16, 17, 18}));
}

TEST_F(ExecutorTest, FailureStopsLaterDumpBatchesAndReleasesSlots)
{
    FakeTransferEndpoint store(8);
    store.fail_first_store_ = true;
    std::vector<DeviceShard> shards{MakeShard(0), MakeShard(1), MakeShard(2), MakeShard(3)};
    auto created = CreateTestExecutor(store, {3, 5}, 0, 2, 2);
    ASSERT_TRUE(created);
    auto executor = std::move(created).Value();

    auto task = executor->Submit(MakeTask(shards), Operation::DUMP);
    ASSERT_TRUE(task);
    EXPECT_TRUE(executor->Wait(task.Value()).Failure());
    EXPECT_EQ(store.StoreBatches(), (std::vector<std::vector<std::size_t>>{
                                          {0, 1}
    }));

    store.fail_first_store_ = false;
    std::vector<DeviceShard> recoveryShards{MakeShard(4), MakeShard(5)};
    auto recovery = executor->Submit(MakeTask(recoveryShards), Operation::DUMP);
    ASSERT_TRUE(recovery);
    EXPECT_TRUE(executor->Wait(recovery.Value()).Success());
    EXPECT_EQ(store.StoreBatches(), (std::vector<std::vector<std::size_t>>{
                                          {0, 1},
                                          {4, 5}
    }));
}

TEST_F(ExecutorTest, DumpTransferGroupsSubmitSeparatelyAndIsolateStoreFailures)
{
    FakeTransferEndpoint store(8);
    store.BlockFirstLoad();
    store.BlockFirstStore();
    store.fail_dump_submit_index_ = 0;
    store.fail_dump_wait_index_ = 1;
    store.SetData(100, {1, 2, 3, 4, 5, 6, 7, 8});
    store.SetData(101, {11, 12, 13, 14, 15, 16, 17, 18});
    store.SetData(102, {21, 22, 23, 24, 25, 26, 27, 28});

    auto created = CreateTestExecutor(store, {3, 5}, 0, 3, 2);
    ASSERT_TRUE(created);
    auto executor = std::move(created).Value();

    std::vector<DeviceShard> blockerShards{
        MakeEmptyShard(100), MakeEmptyShard(101), MakeEmptyShard(102)};
    auto blocker = executor->Submit(MakeTask(blockerShards), Operation::LOAD);
    ASSERT_TRUE(blocker);
    const auto loadCheckSeen = store.WaitForLoadGroupCheck(100);
    if (!loadCheckSeen) { store.ReleaseLoad(); }
    ASSERT_TRUE(loadCheckSeen);

    std::vector<DeviceShard> submitFailedShards{MakeShard(0)};
    std::vector<DeviceShard> waitFailedShards{MakeShard(1)};
    std::vector<DeviceShard> successfulShards{MakeShard(2)};
    auto submitFailed = executor->Submit(MakeTask(submitFailedShards), Operation::DUMP);
    auto waitFailed = executor->Submit(MakeTask(waitFailedShards), Operation::DUMP);
    auto successful = executor->Submit(MakeTask(successfulShards), Operation::DUMP);
    ASSERT_TRUE(submitFailed);
    ASSERT_TRUE(waitFailed);
    ASSERT_TRUE(successful);

    store.ReleaseLoad();
    EXPECT_TRUE(executor->Wait(blocker.Value()).Success());
    const auto storeWaitSeen = store.WaitForStoreWait();
    if (!storeWaitSeen) { store.ReleaseStore(); }
    ASSERT_TRUE(storeWaitSeen);
    EXPECT_EQ(store.StoreBatches(), (std::vector<std::vector<std::size_t>>{
                                          {0},
                                          {1},
                                          {2}
    }));
    store.ReleaseStore();

    EXPECT_TRUE(executor->Wait(submitFailed.Value()).Failure());
    EXPECT_TRUE(executor->Wait(waitFailed.Value()).Failure());
    EXPECT_TRUE(executor->Wait(successful.Value()).Success());
}

TEST_F(ExecutorTest, LoadRunsBeforeNextDumpBatch)
{
    FakeTransferEndpoint store(8);
    store.BlockFirstStore();
    store.SetData(4, {41, 42, 43, 44, 45, 46, 47, 48});
    std::vector<DeviceShard> dumpShards{MakeShard(0), MakeShard(1), MakeShard(2), MakeShard(3)};
    std::vector<DeviceShard> loadShards{MakeEmptyShard(4)};
    auto created = CreateTestExecutor(store, {3, 5}, 0, 2, 2);
    ASSERT_TRUE(created);
    auto executor = std::move(created).Value();

    auto dump = executor->Submit(MakeTask(dumpShards), Operation::DUMP);
    ASSERT_TRUE(dump);
    ASSERT_TRUE(store.WaitForStoreWait());
    auto load = executor->Submit(MakeTask(loadShards), Operation::LOAD);
    ASSERT_TRUE(load);
    store.ReleaseStore();
    ASSERT_TRUE(executor->Wait(load.Value()).Success());
    ASSERT_TRUE(executor->Wait(dump.Value()).Success());

    const auto calls = store.Calls();
    ASSERT_EQ(calls.size(), std::size_t{3});
    EXPECT_EQ(calls[0].operation, Operation::DUMP);
    EXPECT_EQ(calls[1].operation, Operation::LOAD);
    EXPECT_EQ(calls[2].operation, Operation::DUMP);
}

TEST_F(ExecutorTest, DumpWaitDoesNotBlockLoadWorker)
{
    FakeTransferEndpoint store(8);
    store.BlockFirstStore();
    store.SetData(1, {11, 12, 13, 14, 15, 16, 17, 18});
    std::vector<DeviceShard> dumpShards{MakeShard(0)};
    std::vector<DeviceShard> loadShards{MakeEmptyShard(1)};
    auto created = CreateTestExecutor(store, {3, 5}, 0, 2, 2);
    ASSERT_TRUE(created);
    auto executor = std::move(created).Value();

    auto dump = executor->Submit(MakeTask(dumpShards), Operation::DUMP);
    ASSERT_TRUE(dump);
    const auto storeWaitSeen = store.WaitForStoreWait();
    if (!storeWaitSeen) { store.ReleaseStore(); }
    ASSERT_TRUE(storeWaitSeen);

    auto load = executor->Submit(MakeTask(loadShards), Operation::LOAD);
    bool loadCompleted = false;
    if (load) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < deadline) {
            auto checked = executor->Check(load.Value());
            if (checked && checked.Value()) {
                loadCompleted = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    store.ReleaseStore();
    ASSERT_TRUE(load);
    EXPECT_TRUE(loadCompleted);
    EXPECT_TRUE(executor->Wait(load.Value()).Success());
    EXPECT_TRUE(executor->Wait(dump.Value()).Success());
    EXPECT_EQ(ReadShard(loadShards[0]),
              (std::array<std::uint8_t, 8>{11, 12, 13, 14, 15, 16, 17, 18}));
}

TEST_F(ExecutorTest, TaskLargerThanSlotCountCompletesAcrossBatches)
{
    FakeTransferEndpoint store(8);
    std::vector<DeviceShard> shards{MakeShard(0), MakeShard(1)};
    auto created = CreateTestExecutor(store, {3, 5}, 0, 1, 1);
    ASSERT_TRUE(created);
    auto executor = std::move(created).Value();

    auto task = executor->Submit(MakeTask(shards), Operation::DUMP);
    ASSERT_TRUE(task);
    EXPECT_TRUE(executor->Wait(task.Value()).Success());
    EXPECT_EQ(store.StoreBatches(),
              (std::vector<std::vector<std::size_t>>{{0}, {1}}));
}

TEST_F(ExecutorTest, AcceptsTasksWhileAllSlotsAreBusy)
{
    constexpr std::size_t taskCount = 32;
    FakeTransferEndpoint store(8);
    store.BlockFirstStore();
    auto created = CreateTestExecutor(store, {3, 5}, 0, 1, 1);
    ASSERT_TRUE(created);
    auto executor = std::move(created).Value();

    std::vector<std::vector<DeviceShard>> taskShards;
    std::vector<Detail::TaskHandle> handles;
    taskShards.reserve(taskCount);
    handles.reserve(taskCount);

    taskShards.push_back({MakeShard(0)});
    auto first = executor->Submit(MakeTask(taskShards.back()), Operation::DUMP);
    ASSERT_TRUE(first);
    handles.push_back(first.Value());
    const auto storeWaitSeen = store.WaitForStoreWait();
    if (!storeWaitSeen) { store.ReleaseStore(); }
    ASSERT_TRUE(storeWaitSeen);

    bool allSubmitted = true;
    for (std::size_t index = 1; index < taskCount; ++index) {
        taskShards.push_back({MakeShard(index)});
        auto submitted = executor->Submit(MakeTask(taskShards.back()), Operation::DUMP);
        if (!submitted) {
            allSubmitted = false;
            break;
        }
        handles.push_back(submitted.Value());
    }
    store.ReleaseStore();

    ASSERT_TRUE(allSubmitted);
    ASSERT_EQ(handles.size(), taskCount);
    for (const auto handle : handles) {
        EXPECT_TRUE(executor->Wait(handle).Success());
    }
}

TEST_F(ExecutorTest, ShutdownDrainsCurrentBatchAndCancelsLaterBatches)
{
    FakeTransferEndpoint store(8);
    store.BlockFirstStore();
    std::vector<DeviceShard> shards{MakeShard(0), MakeShard(1), MakeShard(2)};
    auto created = CreateTestExecutor(store, {3, 5}, 0, 1, 1);
    ASSERT_TRUE(created);
    auto executor = std::move(created).Value();

    auto task = executor->Submit(MakeTask(shards), Operation::DUMP);
    ASSERT_TRUE(task);
    ASSERT_TRUE(store.WaitForStoreWait());
    std::thread shutdown([&executor]() { executor->Shutdown(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    store.ReleaseStore();
    shutdown.join();

    EXPECT_TRUE(executor->Wait(task.Value()).Failure());
    EXPECT_EQ(store.StoreBatches(), (std::vector<std::vector<std::size_t>>{{0}}));
}

TEST_F(ExecutorTest, CreateRejectsInvalidConfiguration)
{
    FakeTransferEndpoint store(8);

    auto noSlots = CreateTestExecutor(store, {3, 5}, 0, 0);
    ASSERT_FALSE(noSlots);
    EXPECT_EQ(noSlots.Error(), Status::InvalidParam());

    auto noStreams = CreateTestExecutor(store, {3, 5}, 0, 1, 0);
    ASSERT_FALSE(noStreams);
    EXPECT_EQ(noStreams.Error(), Status::InvalidParam());

    auto emptyTensors = CreateTestExecutor(store, {}, 0, 1);
    ASSERT_FALSE(emptyTensors);
    EXPECT_EQ(emptyTensors.Error(), Status::InvalidParam());

    auto zeroTensor = CreateTestExecutor(store, {3, 0}, 0, 1);
    ASSERT_FALSE(zeroTensor);
    EXPECT_EQ(zeroTensor.Error(), Status::InvalidParam());

    auto overflowingTensors =
        CreateTestExecutor(store, {std::numeric_limits<std::size_t>::max(), 1}, 0, 1);
    ASSERT_FALSE(overflowingTensors);
    EXPECT_EQ(overflowingTensors.Error(), Status::InvalidParam());
}

}  // namespace
}  // namespace UC::Delegator
