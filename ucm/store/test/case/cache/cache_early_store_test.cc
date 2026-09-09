// Exercise the exported factory and native transfers, not a separately built buffer.
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include "cache/cc/reserved_prefetch.h"
#include "detail/mock_store.h"
#include "detail/random.h"

extern "C" UC::StoreV1* MakeCacheStore();
extern "C" UC::StoreV1* MakeEarlyCacheStore();

namespace {
class Backend : public testing::NiceMock<UC::Test::Detail::MockStore> {
public:
    std::weak_ptr<UC::ReadLease> lease;
    UC::Expected<std::shared_ptr<UC::ReadLease>> AcquireReadLease(
        const UC::Detail::BlockId*, size_t) override
    {
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
    auto loaded = store->Load({{block, 0, {destination.data()}}});
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
