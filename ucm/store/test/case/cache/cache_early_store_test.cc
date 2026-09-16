// Exercise the exported factory and native transfers, not a separately built buffer.
#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include "cache/cc/reserved_prefetch.h"
#include "cache/cc/load_queue.h"
#include "detail/mock_store.h"
#include "detail/random.h"

extern "C" UC::StoreV1* MakeCacheStore();
extern "C" UC::StoreV1* MakeEarlyCacheStore();

namespace {
class Backend : public testing::NiceMock<UC::Test::Detail::MockStore> {
    std::mutex leaseMutex;
public:
    std::weak_ptr<UC::ReadLease> lease;
    std::vector<size_t> leaseBlocks;
    UC::Expected<std::shared_ptr<UC::ReadLease>> AcquireReadLease(
        const UC::Detail::BlockId*, size_t count) override
    {
        std::lock_guard guard(leaseMutex);
        leaseBlocks.push_back(count);
        auto owned = std::make_shared<UC::ReadLease>();
        lease = owned;
        return owned;
    }
};

UC::Detail::Dictionary Config(UC::StoreV1* backend)
{
    UC::Detail::Dictionary config;
    config.Set("store_backend", backend);
    config.Set("unique_id", UC::Test::Detail::Random().RandomString(16));
    config.SetNumber("device_id", 0);
    config.SetNumber("tensor_size", 4096);
    config.SetNumber("shard_size", 4096);
    config.SetNumber("block_size", 4096);
    config.SetNumber("cache_buffer_capacity_gb", 1);
    config.SetNumber("cache_load_exclusive_buffer_number", 0);
    config.SetNumber("waiting_queue_depth", 8);
    config.SetNumber("running_queue_depth", 16);
    config.SetNumber("cache_stream_number", 1);
    config.Set("share_buffer_enable", false);
    config.Set("io_direct", false);
    return config;
}

UC::CacheStore::Config BatchConfig(UC::StoreV1* backend)
{
    UC::CacheStore::Config config;
    config.uniqueId = UC::Test::Detail::Random().RandomString(16);
    config.deviceId = 0;
    config.tensorSizes = {4096};
    config.shardSize = config.blockSize = 4096;
    config.bufferCapacity = 256 * config.shardSize;
    config.loadExclusiveBufferNumber = 0;
    config.shareBufferEnable = false;
    config.storeBackend = backend;
    return config;
}
}  // namespace

TEST(UCCacheEarlyStoreTest, BatchPinPreservesNativeDataAndStatesAndCoalescesBuckets)
{
    Backend backend;
    auto config = BatchConfig(&backend);
    config.blockSize *= 4;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    auto* buffer = manager.GetTransBuffer();
    std::array<UC::Detail::BlockId, 12> blocks{};
    std::vector<UC::CacheStore::TransBuffer::Handle> owners;
    for (size_t b = 0; b < 6; ++b) {
        blocks[b][0] = std::byte(b);
        blocks[b + 6] = blocks[b];  // exercise identical queries in a bucket
        for (size_t shard = 0; shard < 3; ++shard) {
            auto pin = buffer->Get(blocks[b], shard);
            if (shard == 0) { pin.MarkReady(); }
            if (shard == 2) { pin.MarkFailed(UC::Status::Error("test failure")); }
            owners.push_back(std::move(pin));
        }
    }
    auto batch = UC::CacheStore::EarlyBufferPins::PinBlocks(*buffer, blocks.data(), blocks.size(), 4);
    ASSERT_FALSE(batch.capacityConflict);
    ASSERT_LE(batch.bucketLocks, 24);
    for (size_t q = 0; q < batch.pins.size(); ++q) {
        const auto shard = q % 4;
        EXPECT_EQ(shard < 2, bool(batch.pins[q]));  // READY/live LOADING only
        if (batch.pins[q]) {
            auto& native = owners[(q / 4 % 6) * 3 + shard];
            EXPECT_EQ(native.Data(), batch.pins[q].Data());
            EXPECT_EQ(native.GetState(), batch.pins[q].GetState());
            EXPECT_FALSE(batch.pins[q].Owner());
        }
    }
    for (auto& pin : owners) { pin.MarkReady(); }
    owners.clear();
    EXPECT_TRUE(batch.pins[1].Ready());
    EXPECT_TRUE(batch.pins[25].Ready());
}

TEST(UCCacheEarlyStoreTest, BatchStepSubmitsAnotherRequestWhileFirstReadIsPending)
{
    Backend backend;
    auto c = BatchConfig(&backend);
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(c).Success());
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, c.blockSize, c.shardSize, 1000);
    UC::Detail::BlockId first{}, second{};
    second[0] = std::byte{9};
    EXPECT_CALL(backend, LookupOnPrefix).WillRepeatedly(testing::Invoke(
        [](const UC::Detail::BlockId*, size_t) { return ssize_t{0}; }));
    auto a = early.Reserve(&first, 1).Value();
    auto b = early.Reserve(&second, 1).Value();
    size_t submitted = 0, waited = 0;
    bool complete = false;
    EXPECT_CALL(backend, Load).Times(2).WillRepeatedly(testing::Invoke(
        [&](UC::Detail::TaskDesc task) {
            std::memset(task[0].addrs[0], int(0x40 + ++submitted), c.shardSize);
            return UC::Detail::TaskHandle(submitted);
        }));
    EXPECT_CALL(backend, Check).WillRepeatedly(testing::Invoke(
        [&](auto) { return complete; }));
    EXPECT_CALL(backend, Wait).Times(2).WillRepeatedly(testing::Invoke(
        [&](auto) { ++waited; return UC::Status::OK(); }));
    auto pa = a.batch->Step(c.shardSize).Value();
    EXPECT_FALSE(pa.done);
    EXPECT_EQ(pa.inflightBytes, c.shardSize);
    EXPECT_FALSE(a.batch->Step(0).Value().done);
    auto pb = b.batch->Step(c.shardSize).Value();
    EXPECT_EQ(submitted, 2);  // a's unfinished task did not monopolize submission
    EXPECT_EQ(waited, 0);
    EXPECT_FALSE(pb.done);
    complete = true;
    EXPECT_TRUE(b.batch->Step(0).Value().done);
    EXPECT_TRUE(a.batch->Step(0).Value().done);
    UC::Detail::TaskDesc task{{first, 0, {nullptr}}, {second, 0, {nullptr}}};
    task.loadReservations = {a.handle, b.handle};
    auto pins = early.LoadHandles(task);
    ASSERT_EQ(pins.size(), 2);
    early.Release(a.handle);
    early.Release(b.handle);
    EXPECT_EQ(static_cast<unsigned char*>(pins[0].Data())[0], 0x41);
    EXPECT_EQ(static_cast<unsigned char*>(pins[1].Data())[0], 0x42);
}

TEST(UCCacheEarlyStoreTest, CancelledBatchStopsSubmissionButDrainsBeforeLeaseAndPinRelease)
{
    Backend backend;
    auto c = BatchConfig(&backend);
    c.blockSize *= 128;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(c).Success());
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, c.blockSize, c.shardSize, 1000);
    UC::Detail::BlockId block{};
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(ssize_t{0}));
    auto reservation = early.Reserve(&block, 1).Value();
    bool complete = false;
    EXPECT_CALL(backend, Load).Times(1).WillOnce(testing::Return(UC::Detail::TaskHandle{17}));
    EXPECT_CALL(backend, Check).WillRepeatedly(testing::Invoke([&](auto) { return complete; }));
    EXPECT_CALL(backend, Wait).Times(1).WillOnce(testing::Invoke([&](auto) {
        EXPECT_FALSE(backend.lease.expired());
        return UC::Status::OK();
    }));
    auto first = reservation.batch->Step(64 * c.shardSize).Value();
    EXPECT_EQ(first.inflightBytes, 64 * c.shardSize);
    EXPECT_EQ(reservation.batch->Snapshot().completedReadBytes, 0);
    early.Release(reservation.handle);
    EXPECT_FALSE(backend.lease.expired());
    auto pending = reservation.batch->Step(64 * c.shardSize);
    ASSERT_TRUE(pending);
    EXPECT_FALSE(pending.Value().done);
    complete = true;
    EXPECT_FALSE(reservation.batch->Step(0));
    EXPECT_EQ(reservation.batch->Snapshot().result.l2ReadBytes, 64 * c.shardSize);
    EXPECT_EQ(reservation.batch->Snapshot().completedReadBytes, 64 * c.shardSize);
    EXPECT_TRUE(backend.lease.expired());
}

TEST(UCCacheEarlyStoreTest, AbandonedBatchDrainsAcceptedNativeTask)
{
    Backend backend;
    auto c = BatchConfig(&backend);
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(c).Success());
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, c.blockSize, c.shardSize, 1000);
    UC::Detail::BlockId block{};
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(ssize_t{0}));
    auto reservation = early.Reserve(&block, 1).Value();
    EXPECT_CALL(backend, Load).WillOnce(testing::Return(UC::Detail::TaskHandle{18}));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Invoke([&](auto) {
        EXPECT_FALSE(backend.lease.expired());
        return UC::Status::OK();
    }));
    EXPECT_FALSE(reservation.batch->Step(c.shardSize).Value().done);
    early.Release(reservation.handle);
    EXPECT_FALSE(backend.lease.expired());
    reservation.batch.reset();
    EXPECT_TRUE(backend.lease.expired());
}

TEST(UCCacheEarlyStoreTest, BatchRechecksMissingPositionAndDoesNotReadNewlyReadyData)
{
    Backend backend;
    auto c = BatchConfig(&backend);
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(c).Success());
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, c.blockSize, c.shardSize, 1000);
    UC::Detail::BlockId block{};
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(ssize_t{0}));
    auto reservation = early.Reserve(&block, 1).Value();
    EXPECT_EQ(reservation.remainingL2Work, 1);
    // Another consumer/native dump fills the position between reserve and Step.
    auto native = manager.GetTransBuffer()->Get(block, 0);
    native.MarkReady();
    native = {};
    EXPECT_CALL(backend, Load).Times(0);
    EXPECT_CALL(backend, Wait).Times(0);
    auto result = reservation.batch->Step(c.shardSize).Value();
    EXPECT_TRUE(result.done);
    EXPECT_EQ(result.result.l2ReadBytes, 0);
    early.Release(reservation.handle);
}

TEST(UCCacheEarlyStoreTest, ConcurrentLocalizationSharesOneBlockClaimUntilLastConsumer)
{
    Backend backend;
    auto c = BatchConfig(&backend);
    c.blockSize *= 64;
    c.bufferCapacity = c.blockSize;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(c).Success());
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, c.blockSize, c.shardSize, 1000);
    UC::Detail::BlockId block{}, other{};
    other[0] = std::byte{77};
    for (size_t shard = 0; shard < 64; ++shard) {
        auto pin = manager.GetTransBuffer()->Get(block, shard);
        pin.MarkReady();
    }
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::vector<std::future<UC::Expected<UC::PrefixReservation>>> pending;
    for (size_t i = 0; i < 8; ++i) {
        pending.push_back(std::async(std::launch::async, [&, gate] {
            gate.wait();
            return early.Reserve(&block, 1);
        }));
    }
    start.set_value();
    std::vector<UC::PrefixReservation> consumers;
    for (auto& future : pending) {
        ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
        auto result = future.get();
        ASSERT_TRUE(result);
        EXPECT_EQ(result.Value().remainingL2Work, 0);
        EXPECT_EQ(result.Value().batch->ProtectedBytes(), c.blockSize);
        consumers.push_back(std::move(result.Value()));
    }
    // No request-count limit: eight consumers share one block's byte claim.
    auto blocked = early.Reserve(&other, 1);
    ASSERT_FALSE(blocked);
    EXPECT_EQ(blocked.Error(), UC::Status::NoSpace());
    for (size_t i = 0; i + 1 < consumers.size(); ++i) {
        early.Release(consumers[i].handle);
        EXPECT_FALSE(manager.GetTransBuffer()->TryGet(other, 0));
        EXPECT_EQ(consumers.back().batch->ProtectedBytes(), c.blockSize);
    }
    early.Release(consumers.back().handle);
    EXPECT_EQ(consumers.back().batch->ProtectedBytes(), 0);
    EXPECT_TRUE(manager.GetTransBuffer()->TryGet(other, 0));
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(ssize_t{0}));
    // A failed localization leaves no unresolved promise or byte claim behind.
    auto retry = early.Reserve(&other, 1);
    ASSERT_TRUE(retry);
    early.Release(retry.Value().handle);
}

TEST(UCCacheEarlyStoreTest, OppositeBlockOrderCannotCycleLocalizationWaits)
{
    Backend backend;
    auto c = BatchConfig(&backend);
    c.blockSize *= 64;
    c.bufferCapacity = 2 * c.blockSize;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(c).Success());
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, c.blockSize, c.shardSize, 1000);
    std::array<UC::Detail::BlockId, 2> blocks{};
    blocks[1][0] = std::byte{31};
    for (const auto& block : blocks) {
        for (size_t shard = 0; shard < 64; ++shard) {
            auto pin = manager.GetTransBuffer()->Get(block, shard);
            pin.MarkReady();
        }
    }
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::vector<std::future<UC::Expected<UC::PrefixReservation>>> pending;
    for (size_t i = 0; i < 8; ++i) {
        pending.push_back(std::async(std::launch::async, [&, gate, i] {
            auto ordered = blocks;
            if (i % 2) { std::swap(ordered[0], ordered[1]); }
            gate.wait();
            return early.Reserve(ordered.data(), ordered.size());
        }));
    }
    start.set_value();
    std::vector<UC::PrefixReservation> results;
    for (auto& future : pending) {
        ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
        auto result = future.get();
        ASSERT_TRUE(result);
        results.push_back(std::move(result.Value()));
    }
    for (auto& result : results) { early.Release(result.handle); }
    EXPECT_EQ(results.front().batch->ProtectedBytes(), 0);
}

TEST(UCCacheEarlyStoreTest, OneCandidateBatchUsesOneNativeL2PrefixQuery)
{
    Backend backend;
    auto config = BatchConfig(&backend);
    config.shardSize = 64;
    config.blockSize = 78 * config.shardSize;
    config.bufferCapacity = 512 * config.blockSize;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    std::vector<UC::Detail::BlockId> blocks(512);
    for (size_t i = 0; i < blocks.size(); ++i) { std::memcpy(blocks[i].data(), &i, sizeof(i)); }
    std::vector<size_t> queries;
    EXPECT_CALL(backend, LookupOnPrefix).Times(1).WillRepeatedly(testing::Invoke(
        [&](const UC::Detail::BlockId*, size_t n) { queries.push_back(n); return ssize_t(n) - 1; }));
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, config.blockSize, config.shardSize, 1000);
    auto result = early.Reserve(blocks.data(), blocks.size());
    ASSERT_TRUE(result);
    EXPECT_EQ(result.Value().totalHitBlocks, 512);
    EXPECT_EQ(result.Value().remainingL2Work, 512 * 78);
    EXPECT_EQ(queries, (std::vector<size_t>{512}));
    early.Release(result.Value().handle);
}

TEST(UCCacheEarlyStoreTest, ReusedReadyBlockLivesUntilLastConsumerAndExpiresBeforeEviction)
{
    Backend backend;
    auto c = BatchConfig(&backend);
    c.blockSize *= 2;
    c.bufferCapacity = c.blockSize;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(c).Success());
    UC::Detail::BlockId block{}, other{};
    other[0] = std::byte{17};
    for (size_t shard = 0; shard < 2; ++shard) {
        auto pin = manager.GetTransBuffer()->Get(block, shard);
        std::memset(pin.Data(), 0x5b, c.shardSize);
        pin.MarkReady();
    }
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, c.blockSize, c.shardSize, 1000);
    auto first = early.Reserve(&block, 1).Value();
    auto second = early.Reserve(&block, 1).Value();
    EXPECT_EQ(second.remainingL2Work, 0);
    early.Release(first.handle);
    EXPECT_FALSE(manager.GetTransBuffer()->TryGet(other, 0));
    UC::Detail::TaskDesc task{{block, 1, {nullptr}}};
    task.loadReservations = {second.handle};
    auto handles = early.LoadHandles(task);
    ASSERT_EQ(handles.size(), 1);
    EXPECT_EQ(static_cast<unsigned char*>(handles[0].Data())[0], 0x5b);
    early.Release(second.handle);
    handles.clear();
    std::vector<UC::CacheStore::TransBuffer::Handle> replacement;
    for (size_t shard = 0; shard < 2; ++shard) {
        auto pin = manager.GetTransBuffer()->TryGet(other, shard);
        ASSERT_TRUE(pin);
        pin.MarkReady();
        replacement.push_back(std::move(pin));
    }
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(ssize_t{-1}));
    auto expired = early.Reserve(&block, 1).Value();
    EXPECT_EQ(expired.totalHitBlocks, 0);
    early.Release(expired.handle);
}

TEST(UCCacheEarlyStoreTest, LocationHintsRevalidateIdentityStateAndReleasePartialAcquisitions)
{
    for (bool shared : {false, true}) {
        Backend backend;
        auto c = BatchConfig(&backend);
        c.blockSize *= 2;
        c.bufferCapacity = c.blockSize;
        c.shareBufferEnable = shared;
        UC::CacheStore::BufferManager manager;
        ASSERT_TRUE(manager.Setup(c).Success());
        auto* buffer = manager.GetTransBuffer();
        UC::Detail::BlockId block{}, other{};
        other[0] = std::byte{27};
        std::vector<UC::CacheStore::TransBuffer::Handle> seed;
        for (size_t shard = 0; shard < 2; ++shard) {
            auto pin = buffer->Get(block, shard);
            std::memset(pin.Data(), 0x60 + shard, c.shardSize);
            pin.MarkReady();
            seed.push_back(std::move(pin));
        }
        auto positions = UC::CacheStore::EarlyBufferPins::Locations(seed);
        seed.clear();  // snapshots alone must hold no native reference
        size_t checks = 0;
        auto recalled = UC::CacheStore::EarlyBufferPins::Recall(*buffer, block, positions, checks);
        ASSERT_EQ(recalled.size(), 2);
        EXPECT_EQ(checks, 2);
        for (size_t shard = 0; shard < 2; ++shard) {
            EXPECT_FALSE(recalled[shard].Owner());
            EXPECT_EQ(static_cast<unsigned char*>(recalled[shard].Data())[0], 0x60 + shard);
        }
        recalled.clear();
        EXPECT_TRUE(UC::CacheStore::EarlyBufferPins::Recall(*buffer, other, positions, checks).empty());
        auto wrongShard = positions;
        std::swap(wrongShard[0], wrongShard[1]);
        EXPECT_TRUE(UC::CacheStore::EarlyBufferPins::Recall(*buffer, block, wrongShard, checks).empty());
        auto invalid = positions;
        invalid[1] = std::numeric_limits<size_t>::max();
        EXPECT_TRUE(UC::CacheStore::EarlyBufferPins::Recall(*buffer, block, invalid, checks).empty());
        {
            auto failed = buffer->Get(block, 1);
            failed.MarkFailed(UC::Status::Error("test invalidated second shard"));
        }
        checks = 0;
        EXPECT_TRUE(UC::CacheStore::EarlyBufferPins::Recall(*buffer, block, positions, checks).empty());
        EXPECT_EQ(checks, 2);  // shard 0 was acquired before shard 1 failed validation
        std::vector<UC::CacheStore::TransBuffer::Handle> replacement;
        for (size_t shard = 0; shard < 2; ++shard) {
            auto pin = buffer->TryGet(other, shard);
            ASSERT_TRUE(pin);  // both native slots are free, including the partial acquisition
            pin.MarkReady();
            replacement.push_back(std::move(pin));
        }
        EXPECT_TRUE(UC::CacheStore::EarlyBufferPins::Recall(*buffer, block, positions, checks).empty());
    }
}

TEST(UCCacheEarlyStoreTest, LocationHintsStaySafeDuringNativeEviction)
{
    for (bool shared : {false, true}) {
        Backend backend;
        auto c = BatchConfig(&backend);
        c.blockSize *= 2;
        c.bufferCapacity = 2 * c.blockSize;
        c.shareBufferEnable = shared;
        UC::CacheStore::BufferManager manager;
        ASSERT_TRUE(manager.Setup(c).Success());
        auto* buffer = manager.GetTransBuffer();
        UC::Detail::BlockId block{};
        std::vector<UC::CacheStore::TransBuffer::Handle> seed;
        for (size_t shard = 0; shard < 2; ++shard) {
            auto pin = buffer->Get(block, shard);
            std::memset(pin.Data(), 0x60 + shard, c.shardSize);
            pin.MarkReady();
            seed.push_back(std::move(pin));
        }
        auto positions = UC::CacheStore::EarlyBufferPins::Locations(seed);
        seed.clear();
        auto eviction = std::async(std::launch::async, [&] {
            for (size_t i = 1; i < 512; ++i) {
                UC::Detail::BlockId other{};
                std::memcpy(other.data(), &i, sizeof(i));
                auto pin = buffer->TryGet(other, i % 2);
                if (pin) {
                    std::memset(pin.Data(), 0x23, c.shardSize);
                    pin.MarkReady();
                }
            }
        });
        for (size_t i = 0; i < 512; ++i) {
            size_t checks = 0;
            auto pins = UC::CacheStore::EarlyBufferPins::Recall(*buffer, block, positions, checks);
            for (size_t shard = 0; shard < pins.size(); ++shard) {
                EXPECT_EQ(static_cast<unsigned char*>(pins[shard].Data())[0], 0x60 + shard);
            }
        }
        eviction.get();
        std::vector<UC::CacheStore::TransBuffer::Handle> allCapacity;
        for (size_t shard = 0; shard < 4; ++shard) {
            auto pin = buffer->TryGet(block, shard);
            ASSERT_TRUE(pin);
            allCapacity.push_back(std::move(pin));
        }
    }
}

TEST(UCCacheEarlyStoreTest, SequentialReleasedReservationsRetainOnlyDisposableHints)
{
    Backend backend;
    auto c = BatchConfig(&backend);
    c.blockSize *= 2;
    c.bufferCapacity = c.blockSize;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(c).Success());
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, c.blockSize, c.shardSize, 1000);
    EXPECT_CALL(backend, LookupOnPrefix).Times(0);
    EXPECT_CALL(backend, Load).Times(0);
    for (size_t generation = 1; generation < 32; ++generation) {
        UC::Detail::BlockId block{};
        block[0] = std::byte(generation);
        for (size_t shard = 0; shard < 2; ++shard) {
            auto pin = manager.GetTransBuffer()->TryGet(block, shard);
            ASSERT_TRUE(pin);  // previous generations retain no protection or capacity
            std::memset(pin.Data(), int(generation), c.shardSize);
            pin.MarkReady();
        }
        for (size_t repeat = 0; repeat < 3; ++repeat) {
            auto result = early.Reserve(&block, 1);
            ASSERT_TRUE(result);
            EXPECT_EQ(result.Value().totalHitBlocks, 1);
            EXPECT_EQ(result.Value().remainingL2Work, 0);
            UC::Detail::TaskDesc task{{block, 1, {nullptr}}};
            task.loadReservations = {result.Value().handle};
            auto handles = early.LoadHandles(task);
            early.Release(result.Value().handle);
            ASSERT_EQ(handles.size(), 1);
            EXPECT_EQ(static_cast<unsigned char*>(handles[0].Data())[0], generation);
        }
    }
}

TEST(UCCacheEarlyStoreTest, LoadPlanIsScopedToExplicitReservationsAndInvalidatedOnRelease)
{
    Backend backend;
    auto c = BatchConfig(&backend);
    c.blockSize *= 2;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(c).Success());
    std::array<UC::Detail::BlockId, 16> blocks{};
    for (size_t b = 0; b < blocks.size(); ++b) {
        blocks[b][0] = std::byte(b);
        for (size_t shard = 0; shard < 2; ++shard) {
            auto pin = manager.GetTransBuffer()->Get(blocks[b], shard);
            std::memset(pin.Data(), int(b), c.shardSize);
            pin.MarkReady();
        }
    }
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, c.blockSize, c.shardSize, 1000);
    UC::Detail::TaskDesc task;
    for (const auto& block : blocks) { task.loadReservations.push_back(early.Reserve(&block, 1).Value().handle); }
    // Revisit non-adjacent blocks and reverse the shard order.
    for (size_t shard : {size_t(1), size_t(0)}) {
        for (const auto& block : blocks) { task.push_back({block, shard, {nullptr}}); }
    }
    for (size_t layer = 0; layer < 3; ++layer) {
        auto handles = early.LoadHandles(task);
        ASSERT_EQ(handles.size(), task.size());
        for (size_t i = 0; i < task.size(); ++i) {
            ASSERT_TRUE(handles[i]);
            EXPECT_EQ(static_cast<unsigned char*>(handles[i].Data())[0], i % 16);
        }
    }
    auto sameBlock = early.Reserve(&blocks[0], 1).Value().handle;
    early.Release(task.loadReservations[0]);
    auto cancelled = early.LoadHandles(task);
    ASSERT_EQ(cancelled.size(), task.size());
    EXPECT_FALSE(cancelled[0]);
    EXPECT_FALSE(cancelled[16]);  // another consumer is not authorization for this load
    task.loadReservations[0] = sameBlock;
    auto rebound = early.LoadHandles(task);
    EXPECT_TRUE(rebound[0]);
    for (auto handle : task.loadReservations) { early.Release(handle); }
    EXPECT_TRUE(early.LoadHandles(task).empty());
}

TEST(UCCacheEarlyStoreTest, PendingLoadPlanCannotHideLaterReadyTransition)
{
    Backend backend;
    auto c = BatchConfig(&backend);
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(c).Success());
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(ssize_t{0}));
    EXPECT_CALL(backend, Load).WillOnce(testing::Return(UC::Detail::TaskHandle{1}));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::OK()));
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, c.blockSize, c.shardSize, 1000);
    UC::Detail::BlockId block{};
    auto r = early.Reserve(&block, 1).Value();
    UC::Detail::TaskDesc task{{block, 0, {nullptr}}};
    task.loadReservations = {r.handle};
    EXPECT_TRUE(early.LoadHandles(task).empty());
    ASSERT_TRUE(early.Prefetch(r.handle));
    auto handles = early.LoadHandles(task);
    ASSERT_EQ(handles.size(), 1);
    EXPECT_TRUE(handles[0].Ready());
    early.Release(r.handle);
}

TEST(UCCacheEarlyStoreTest, ConcurrentConsumersShareReadyProtectionWithoutOwningIo)
{
    Backend backend;
    auto c = BatchConfig(&backend);
    c.blockSize *= 2;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(c).Success());
    UC::Detail::BlockId block{};
    for (size_t shard = 0; shard < 2; ++shard) {
        auto pin = manager.GetTransBuffer()->Get(block, shard);
        pin.MarkReady();
    }
    EXPECT_CALL(backend, LookupOnPrefix).Times(0);
    EXPECT_CALL(backend, Load).Times(0);
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, c.blockSize, c.shardSize, 1000);
    auto held = early.Reserve(&block, 1).Value();
    std::vector<std::future<bool>> readers;
    for (size_t thread = 0; thread < 4; ++thread) {
        readers.push_back(std::async(std::launch::async, [&] {
            for (size_t i = 0; i < 8; ++i) {
                auto r = early.Reserve(&block, 1);
                if (!r || r.Value().remainingL2Work != 0) { return false; }
                auto done = early.Prefetch(r.Value().handle);
                if (!done || done.Value().l2ReadShards != 0) { return false; }
                early.Release(r.Value().handle);
            }
            return true;
        }));
    }
    for (auto& reader : readers) { EXPECT_TRUE(reader.get()); }
    early.Release(held.handle);
}

TEST(UCCacheEarlyStoreTest, ReservedBytesUseNativeLoadAndRelease)
{
    Backend backend;
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(ssize_t{0}));
    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([](UC::Detail::TaskDesc task) {
        for (auto& shard : task) { std::memset(shard.addrs[0], 0x6b, 4096); }
        return UC::Detail::TaskHandle{1};
    }));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::OK()));
    std::unique_ptr<UC::StoreV1> store(MakeEarlyCacheStore());
    ASSERT_TRUE(store->Setup(Config(&backend)).Success());
    UC::Detail::BlockId block{};
    auto reserved = store->ReservePrefix(&block, 1);
    ASSERT_TRUE(reserved);
    EXPECT_EQ(reserved.Value().remainingL2Work, 1);
    ASSERT_FALSE(backend.lease.expired());
    auto prefetched = store->PrefetchReserved(reserved.Value().handle);
    ASSERT_TRUE(prefetched);
    EXPECT_EQ(prefetched.Value().l2ReadShards, 1);
    std::array<unsigned char, 4096> destination{};
    UC::Detail::TaskDesc desc{{block, 0, {destination.data()}}};
    desc.loadReservations = {reserved.Value().handle};
    auto loaded = store->Load(std::move(desc));
    ASSERT_TRUE(loaded);
    ASSERT_TRUE(store->Wait(loaded.Value()).Success());
    for (auto byte : destination) { ASSERT_EQ(byte, 0x6b); }
    store->ReleaseReservation(reserved.Value().handle);
    EXPECT_TRUE(backend.lease.expired());
    store->ReleaseReservation(reserved.Value().handle);
    // A second reservation sees the exact buffer filled by the first one.
    auto ready = store->ReservePrefix(&block, 1);
    ASSERT_TRUE(ready);
    EXPECT_EQ(ready.Value().remainingL2Work, 0);
    EXPECT_EQ(ready.Value().totalHitBlocks, 1);
    store.reset();
    EXPECT_TRUE(backend.lease.expired());
}

TEST(UCCacheEarlyStoreTest, NativeFactoryRemainsNative)
{
    Backend backend;
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(ssize_t{-1}));
    UC::Detail::BlockId block{};
    std::unique_ptr<UC::StoreV1> store(MakeCacheStore());
    ASSERT_TRUE(store->Setup(Config(&backend)).Success());
    EXPECT_FALSE(store->ReservePrefix(&block, 1));
    EXPECT_FALSE(store->PrefetchReserved(1));
    EXPECT_EQ(store->LookupOnPrefix(&block, 1).Value(), -1);
    EXPECT_TRUE(backend.lease.expired());
}

TEST(UCCacheEarlyStoreTest, EarlyFactoryPreservesNativeSetupErrors)
{
    Backend backend;
    UC::Detail::BlockId block{};
    std::unique_ptr<UC::StoreV1> invalid(MakeEarlyCacheStore());
    auto config = Config(&backend);
    config.SetNumber("shard_size", 0);
    EXPECT_TRUE(invalid->Setup(config).Failure());
    EXPECT_FALSE(invalid->ReservePrefix(&block, 1));
}

TEST(UCCacheEarlyStoreTest, PartialL1RequiresReadableL2AndKeepsOnlyContinuousPrefix)
{
    Backend backend;
    UC::CacheStore::Config config;
    config.uniqueId = UC::Test::Detail::Random().RandomString(16);
    config.deviceId = 0;
    config.shardSize = 4096;
    config.blockSize = 8192;
    config.bufferCapacity = 4 << 20;
    config.loadExclusiveBufferNumber = 0;
    config.shareBufferEnable = false;
    config.storeBackend = &backend;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    auto* buffer = manager.GetTransBuffer();
    std::array<UC::Detail::BlockId, 4> blocks{};
    for (size_t i = 0; i < blocks.size(); ++i) { blocks[i][0] = std::byte(i); }
    for (auto i : {0, 3}) {
        for (size_t shard = 0; shard < 2; ++shard) {
            auto pin = buffer->Get(blocks[i], shard);
            pin.MarkReady();
        }
    }
    auto partial = buffer->Get(blocks[1], 0);
    partial.MarkReady();
    auto pending = buffer->Get(blocks[2], 0);
    // Native lookup accepts even an allocated, non-READY first shard.
    ASSERT_EQ(manager.LookupOnPrefix(blocks.data(), blocks.size()).Value(), 3);
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Invoke(
        [&](const UC::Detail::BlockId* needed, size_t count) {
            EXPECT_EQ(count, 2);
            EXPECT_EQ(needed[0], blocks[1]);
            EXPECT_EQ(needed[1], blocks[2]);
            return ssize_t{0};  // Only the partial block has readable L2.
        }));
    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([](UC::Detail::TaskDesc task) {
        EXPECT_EQ(task.size(), 1);
        for (auto& shard : task) { std::memset(shard.addrs[0], 0x6b, 4096); }
        return UC::Detail::TaskHandle{1};
    }));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::OK()));
    UC::CacheStore::ReservedPrefetch early(
        &manager, &backend, config.blockSize, config.shardSize, config.timeoutMs);
    auto reserved = early.Reserve(blocks.data(), blocks.size());
    ASSERT_TRUE(reserved);
    EXPECT_EQ(reserved.Value().totalHitBlocks, 2);
    EXPECT_EQ(reserved.Value().remainingL2Work, 1);
    auto copied = early.Prefetch(reserved.Value().handle);
    ASSERT_TRUE(copied);
    EXPECT_EQ(copied.Value().l2ReadShards, 1);
    early.Release(reserved.Value().handle);
    EXPECT_TRUE(backend.lease.expired());
}

TEST(UCCacheEarlyStoreTest, ReadyShortPrefixHasNoPinsBeyondThePublishedHit)
{
    Backend backend;
    auto config = BatchConfig(&backend);
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    auto* buffer = manager.GetTransBuffer();
    std::array<UC::Detail::BlockId, 4> blocks{};
    for (size_t i = 0; i < blocks.size(); ++i) { blocks[i][0] = std::byte(i); }
    for (auto i : {0, 2, 3}) {
        auto pin = buffer->Get(blocks[i], 0);
        pin.MarkReady();
    }
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(ssize_t{-1}));
    EXPECT_CALL(backend, Load).Times(0);
    UC::CacheStore::ReservedPrefetch early(
        &manager, &backend, config.blockSize, config.shardSize, config.timeoutMs);
    auto reserved = early.Reserve(blocks.data(), blocks.size());
    ASSERT_TRUE(reserved);
    ASSERT_EQ(reserved.Value().totalHitBlocks, 1);
    ASSERT_EQ(reserved.Value().remainingL2Work, 0);
    // Native Owner() means this Get acquired the first live reference. The
    // broker may refund this tail, but must retain the protected first block.
    EXPECT_FALSE(buffer->Get(blocks[0], 0).Owner());
    EXPECT_TRUE(buffer->Get(blocks[2], 0).Owner());
    EXPECT_TRUE(buffer->Get(blocks[3], 0).Owner());
    early.Release(reserved.Value().handle);
    EXPECT_TRUE(buffer->Get(blocks[0], 0).Owner());
}

TEST(UCCacheEarlyStoreTest, BoundedAllocationPreservesPinnedNativeEntries)
{
    Backend backend;
    UC::CacheStore::Config config;
    config.uniqueId = UC::Test::Detail::Random().RandomString(16);
    config.deviceId = 0;
    config.shardSize = config.blockSize = 4096;
    config.bufferCapacity = 4 * config.shardSize;
    config.loadExclusiveBufferNumber = 0;
    config.shareBufferEnable = false;
    config.storeBackend = &backend;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    auto* buffer = manager.GetTransBuffer();
    std::array<UC::CacheStore::TransBuffer::Handle, 4> pins;
    for (size_t i = 0; i < pins.size(); ++i) {
        UC::Detail::BlockId block{};
        block[0] = std::byte(i);
        pins[i] = buffer->Get(block, 0);
        ASSERT_TRUE(pins[i]);
        std::memset(pins[i].Data(), 0x30 + i, config.shardSize);
        pins[i].MarkReady();
    }
    UC::Detail::BlockId missing{};
    missing[0] = std::byte{9};
    auto attempt = std::async(std::launch::async, [&] { return buffer->TryGet(missing, 0); });
    const auto completed = attempt.wait_for(std::chrono::seconds(1));
    // Release capacity even if a regression accidentally uses unbounded Alloc,
    // so that the failing test can join its worker instead of hanging the suite.
    pins[0] = {};
    auto result = attempt.get();
    EXPECT_EQ(completed, std::future_status::ready);
    EXPECT_FALSE(result);
    result = {};
    for (size_t i = 1; i < pins.size(); ++i) {
        EXPECT_TRUE(pins[i].Ready());
        EXPECT_EQ(static_cast<unsigned char*>(pins[i].Data())[0], 0x30 + i);
    }
    auto native = buffer->Get(missing, 0);
    EXPECT_TRUE(native);
    EXPECT_TRUE(native.Owner());
}

TEST(UCCacheEarlyStoreTest, ShortHitStopsCandidateLeaseAndShardWork)
{
    Backend backend;
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Invoke(
        [](const UC::Detail::BlockId*, size_t count) {
            EXPECT_EQ(count, 32);
            return ssize_t{-1};
        }));
    EXPECT_CALL(backend, Load).Times(0);
    std::unique_ptr<UC::StoreV1> store(MakeEarlyCacheStore());
    ASSERT_TRUE(store->Setup(Config(&backend)).Success());
    std::array<UC::Detail::BlockId, 256> blocks{};
    for (size_t i = 0; i < blocks.size(); ++i) { blocks[i][0] = std::byte(i); }
    // EarlyWorker owns adaptive growth and stops after this first miss batch.
    auto reserved = store->ReservePrefix(blocks.data(), 32);
    ASSERT_TRUE(reserved);
    EXPECT_EQ(reserved.Value().totalHitBlocks, 0);
    EXPECT_EQ(reserved.Value().remainingL2Work, 0);
    EXPECT_EQ(backend.leaseBlocks, std::vector<size_t>{32});
    store->ReleaseReservation(reserved.Value().handle);
    EXPECT_TRUE(backend.lease.expired());
}

TEST(UCCacheEarlyStoreTest, InflightPositionStaysPinnedAfterOwnerCompletes)
{
    Backend backend;
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(ssize_t{0}));
    EXPECT_CALL(backend, Load).Times(0);
    UC::CacheStore::Config config;
    config.uniqueId = UC::Test::Detail::Random().RandomString(16);
    config.deviceId = 0;
    config.shardSize = config.blockSize = 4096;
    config.bufferCapacity = 4 * config.shardSize;
    config.loadExclusiveBufferNumber = 0;
    config.shareBufferEnable = false;
    config.storeBackend = &backend;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    auto* buffer = manager.GetTransBuffer();
    UC::Detail::BlockId block{};
    auto owner = buffer->Get(block, 0);
    ASSERT_TRUE(owner.Owner());
    UC::CacheStore::ReservedPrefetch early(
        &manager, &backend, config.blockSize, config.shardSize, config.timeoutMs);
    auto reserved = early.Reserve(&block, 1);
    ASSERT_TRUE(reserved);
    EXPECT_EQ(reserved.Value().remainingL2Work, 1);
    std::memset(owner.Data(), 0x6b, config.shardSize);
    owner.MarkReady();
    owner = {};
    // Churn the same native buffer between reserve and prefetch. A borrowed
    // in-flight position must survive its producer releasing the original pin.
    for (size_t i = 1; i < 64; ++i) {
        UC::Detail::BlockId other{};
        other[0] = std::byte(i);
        auto pin = buffer->Get(other, 0);
        std::memset(pin.Data(), 0x31, config.shardSize);
        pin.MarkReady();
    }
    auto prefetched = early.Prefetch(reserved.Value().handle);
    ASSERT_TRUE(prefetched);
    EXPECT_EQ(prefetched.Value().l2ReadShards, 0);
    auto batch = UC::CacheStore::EarlyBufferPins::PinBlocks(*buffer, &block, 1, 1);
    auto& retained = batch.pins[0];
    ASSERT_TRUE(retained);
    EXPECT_EQ(static_cast<unsigned char*>(retained.Data())[0], 0x6b);
    early.Release(reserved.Value().handle);
    EXPECT_TRUE(backend.lease.expired());
}

TEST(UCCacheEarlyStoreTest, HandoffSurvivesReleaseAndUsesNativeTransferWithoutAnIndex)
{
    Backend backend;
    EXPECT_CALL(backend, LookupOnPrefix).Times(0);
    EXPECT_CALL(backend, Load).Times(0);
    UC::CacheStore::Config config;
    config.uniqueId = UC::Test::Detail::Random().RandomString(16);
    config.deviceId = 0;
    config.tensorSizes = {4096};
    config.shardSize = 4096;
    config.blockSize = 8192;
    config.bufferCapacity = 4 * config.shardSize;
    config.loadExclusiveBufferNumber = 0;
    config.shareBufferEnable = false;
    config.storeBackend = &backend;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    auto* buffer = manager.GetTransBuffer();
    UC::Detail::BlockId block{};
    for (size_t shard = 0; shard < 2; ++shard) {
        auto pin = buffer->Get(block, shard);
        std::memset(pin.Data(), 0x50 + shard, config.shardSize);
        pin.MarkReady();
    }
    UC::CacheStore::ReservedPrefetch early(
        &manager, &backend, config.blockSize, config.shardSize, config.timeoutMs);
    auto reserved = early.Reserve(&block, 1);
    ASSERT_TRUE(reserved);
    ASSERT_EQ(reserved.Value().remainingL2Work, 0);
    std::array<unsigned char, 8192> destination{};
    UC::Detail::TaskDesc desc{
        {block, 1, {destination.data()}}, {block, 0, {destination.data() + 4096}}};
    desc.loadReservations = {reserved.Value().handle};
    auto first = early.LoadHandles(desc);
    auto second = early.LoadHandles(desc);
    ASSERT_EQ(first.size(), 2);
    ASSERT_EQ(second.size(), 2);
    early.Release(reserved.Value().handle);
    EXPECT_TRUE(backend.lease.expired());
    EXPECT_TRUE(early.LoadHandles(desc).empty());  // stale reservation falls back
    first.clear();  // One consumer must not release another consumer's references.
    for (size_t i = 1; i < 64; ++i) {
        UC::Detail::BlockId other{};
        other[0] = std::byte(i);
        auto pin = buffer->Get(other, 0);
        std::memset(pin.Data(), 0x31, config.shardSize);
        pin.MarkReady();
    }
    auto task = std::make_shared<UC::CacheStore::TransTask>(
        UC::CacheStore::TransTask::Type::LOAD, std::move(desc));
    task->loadHandles = std::move(second);
    UC::HashSet<UC::Detail::TaskHandle> failures;
    UC::CacheStore::LoadQueue queue;
    // A task whose handles are all supplied needs neither Get nor Prealloc.
    // Deliberately supply no lookup index to catch either accidental revisit.
    ASSERT_TRUE(queue.Setup(config, &failures, nullptr).Success());
    auto waiter = std::make_shared<UC::Latch>();
    queue.Submit(task, waiter);
    ASSERT_TRUE(waiter->WaitForDuration(2000));
    EXPECT_FALSE(failures.Contains(task->id));
    for (size_t i = 0; i < destination.size(); ++i) {
        ASSERT_EQ(destination[i], i < 4096 ? 0x51 : 0x50);
    }
}

TEST(UCCacheEarlyStoreTest, HandoffIgnoresWrongIdentityAndUnreadyReservations)
{
    Backend backend;
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(ssize_t{0}));
    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([](UC::Detail::TaskDesc task) {
        for (auto& shard : task) { std::memset(shard.addrs[0], 0x6b, 4096); }
        return UC::Detail::TaskHandle{1};
    }));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::OK()));
    UC::CacheStore::Config config;
    config.uniqueId = UC::Test::Detail::Random().RandomString(16);
    config.deviceId = 0;
    config.shardSize = config.blockSize = 4096;
    config.bufferCapacity = 4 * config.shardSize;
    config.loadExclusiveBufferNumber = 0;
    config.shareBufferEnable = false;
    config.storeBackend = &backend;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    UC::CacheStore::ReservedPrefetch early(
        &manager, &backend, config.blockSize, config.shardSize, config.timeoutMs);
    UC::Detail::BlockId block{}, other{};
    other[0] = std::byte{1};
    auto reserved = early.Reserve(&block, 1);
    ASSERT_TRUE(reserved);
    UC::Detail::TaskDesc desc{{block, 0, {}}, {other, 0, {}}, {block, 1, {}}};
    desc.loadReservations = {reserved.Value().handle};
    EXPECT_TRUE(early.LoadHandles(desc).empty());
    ASSERT_TRUE(early.Prefetch(reserved.Value().handle));
    auto handles = early.LoadHandles(desc);
    ASSERT_EQ(handles.size(), 3);
    EXPECT_TRUE(handles[0]);
    EXPECT_FALSE(handles[1]);
    EXPECT_FALSE(handles[2]);
    early.Release(reserved.Value().handle);
    EXPECT_TRUE(early.LoadHandles(desc).empty());
}

TEST(UCCacheEarlyStoreTest, SubmitsBoundedBatchesBeforeLocatingTheTail)
{
    Backend backend;
    EXPECT_CALL(backend, LookupOnPrefix).WillRepeatedly(testing::Invoke(
        [](const UC::Detail::BlockId*, size_t count) { return ssize_t(count) - 1; }));
    auto config = BatchConfig(&backend);
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    auto* buffer = manager.GetTransBuffer();
    std::array<UC::Detail::BlockId, 160> blocks{};
    for (size_t i = 0; i < blocks.size(); ++i) { blocks[i][0] = std::byte(i); }
    size_t submitted = 0, completed = 0, inflight = 0;
    std::vector<size_t> sizes;
    EXPECT_CALL(backend, Load).Times(3).WillRepeatedly(testing::Invoke(
        [&](UC::Detail::TaskDesc task) {
            if (submitted == 0) { EXPECT_FALSE(buffer->Exist(blocks.back(), 0)); }
            sizes.push_back(task.size());
            ++inflight;
            EXPECT_LE(inflight, 2);
            for (auto& shard : task) { std::memset(shard.addrs[0], 0x6b, 4096); }
            return UC::Detail::TaskHandle{++submitted};
        }));
    EXPECT_CALL(backend, Wait).Times(3).WillRepeatedly(testing::Invoke(
        [&](UC::Detail::TaskHandle handle) {
            if (completed == 0) {
                EXPECT_EQ(submitted, 2);  // the next batch overlaps the first I/O
                EXPECT_FALSE(buffer->Exist(blocks.back(), 0));
            }
            EXPECT_EQ(handle, ++completed);
            --inflight;
            return UC::Status::OK();
        }));
    UC::CacheStore::ReservedPrefetch early(
        &manager, &backend, config.blockSize, config.shardSize, config.timeoutMs);
    auto reserved = early.Reserve(blocks.data(), blocks.size());
    ASSERT_TRUE(reserved);
    auto prefetched = early.Prefetch(reserved.Value().handle);
    ASSERT_TRUE(prefetched);
    EXPECT_EQ(sizes, (std::vector<size_t>{64, 64, 32}));
    EXPECT_EQ(prefetched.Value().l2ReadShards, blocks.size());
    EXPECT_EQ(prefetched.Value().l2ReadBytes, blocks.size() * config.shardSize);
    EXPECT_EQ(inflight, 0);
    early.Release(reserved.Value().handle);
    EXPECT_TRUE(backend.lease.expired());
}

class UCCacheEarlyBatchFailureTest : public testing::TestWithParam<int> {};

TEST_P(UCCacheEarlyBatchFailureTest, DrainsAcceptedIoBeforeReleasingProtection)
{
    // 0: second submit fails; 1: first Wait fails; 2: cancellation during Wait;
    // 3: second submit throws before accepting I/O.
    const auto mode = GetParam();
    Backend backend;
    EXPECT_CALL(backend, LookupOnPrefix).WillRepeatedly(testing::Invoke(
        [](const UC::Detail::BlockId*, size_t count) { return ssize_t(count) - 1; }));
    auto config = BatchConfig(&backend);
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    UC::CacheStore::ReservedPrefetch early(
        &manager, &backend, config.blockSize, config.shardSize, config.timeoutMs);
    std::array<UC::Detail::BlockId, 160> blocks{};
    for (size_t i = 0; i < blocks.size(); ++i) { blocks[i][0] = std::byte(i); }
    auto reserved = early.Reserve(blocks.data(), blocks.size());
    ASSERT_TRUE(reserved);
    size_t submitted = 0, completed = 0;
    EXPECT_CALL(backend, Load).Times(2).WillRepeatedly(testing::Invoke(
        [&](UC::Detail::TaskDesc) -> UC::Expected<UC::Detail::TaskHandle> {
            ++submitted;
            if (submitted == 2 && mode == 0) { return UC::Status::NotFound(); }
            if (submitted == 2 && mode == 3) { throw std::runtime_error("submit failed"); }
            return UC::Detail::TaskHandle{submitted};
        }));
    EXPECT_CALL(backend, Wait).Times(mode == 0 || mode == 3 ? 1 : 2)
        .WillRepeatedly(testing::Invoke([&](UC::Detail::TaskHandle handle) {
            EXPECT_FALSE(backend.lease.expired());
            EXPECT_EQ(handle, ++completed);
            if (mode == 2 && completed == 1) {
                early.Release(reserved.Value().handle);
                EXPECT_FALSE(backend.lease.expired());
            }
            return mode == 1 && completed == 1 ? UC::Status::NotFound() : UC::Status::OK();
        }));
    auto prefetched = early.Prefetch(reserved.Value().handle);
    EXPECT_FALSE(prefetched);
    early.Release(reserved.Value().handle);
    EXPECT_TRUE(backend.lease.expired());
    auto tail = UC::CacheStore::EarlyBufferPins::PinBlocks(*manager.GetTransBuffer(), &blocks.back(), 1, 1);
    EXPECT_FALSE(tail.pins[0]);  // later data was neither submitted nor declared READY
}

INSTANTIATE_TEST_SUITE_P(AbnormalExit, UCCacheEarlyBatchFailureTest, testing::Values(0, 1, 2, 3));
