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
public:
    std::weak_ptr<UC::ReadLease> lease;
    std::vector<size_t> leaseBlocks;
    UC::Expected<std::shared_ptr<UC::ReadLease>> AcquireReadLease(
        const UC::Detail::BlockId*, size_t count) override
    {
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
    auto reserved = store->ReservePrefix(blocks.data(), blocks.size());
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
    auto retained = buffer->PinReady(block, 0);
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
    auto pin = manager.GetTransBuffer()->PinReady(blocks.back(), 0);
    EXPECT_FALSE(pin);  // later data was neither submitted nor declared READY
}

INSTANTIATE_TEST_SUITE_P(AbnormalExit, UCCacheEarlyBatchFailureTest, testing::Values(0, 1, 2, 3));
