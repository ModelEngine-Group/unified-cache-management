#include <array>
#include <filesystem>
#include <fstream>
#include "detail/path_base.h"
#include "posix/cc/read_lease.h"

extern "C" UC::StoreV1* MakePosixStore();
extern "C" UC::StoreV1* MakeEarlyPosixStore();

namespace {
using namespace UC;
using namespace UC::PosixStore;
using Detail::BlockId;
class UCEarlyPosixStoreTest : public UC::Test::Detail::PathBase {};

Detail::Dictionary StoreConfig(const std::string& path)
{
    Detail::Dictionary config;
    config.Set("storage_backends", std::vector<std::string>{path});
    config.SetNumber("device_id", 0);
    config.SetNumber("tensor_size", size_t{4096});
    config.SetNumber("shard_size", size_t{4096});
    config.SetNumber("block_size", size_t{4096});
    config.SetNumber("data_dir_shard_bytes", size_t{1});
    config.SetNumber("posix_data_trans_concurrency", size_t{1});
    config.SetNumber("posix_lookup_concurrency", size_t{1});
    config.Set("posix_io_engine", std::string{"psync"});
    config.Set("io_direct", false);
    return config;
}

TEST_F(UCEarlyPosixStoreTest, FactoryUsesNativeIoAndLayoutWithEarlyOnAndOff)
{
    for (bool early : {false, true}) {
        auto path = Path() + (early ? "/on" : "/off");
        std::filesystem::create_directory(path);
        std::unique_ptr<StoreV1> store{
            early ? MakeEarlyPosixStore() : MakePosixStore()};
        ASSERT_EQ(store->Setup(StoreConfig(path)), Status::OK());
        EXPECT_EQ(store->CheckHealth(), Status::OK());
        BlockId block{};
        std::array<char, 4096> source{}, target{};
        source.fill('k');
        Detail::TaskDesc dump;
        dump.push_back(Detail::Shard{block, 0, {source.data()}});
        auto task = store->Dump(std::move(dump));
        ASSERT_TRUE(task);
        EXPECT_EQ(store->Wait(task.Value()), Status::OK());
        auto hit = store->LookupOnPrefix(&block, 1);
        ASSERT_TRUE(hit);
        EXPECT_EQ(hit.Value(), 0);

        // This independent observer uses the native layout, including sharding.
        Config config;
        config.storageBackends = {path};
        config.dataDirShardBytes = 1;
        config.acquireRemoveGuard = early ? PosixReadLease::AcquireRemoveGuard : nullptr;
        SpaceLayout observer;
        ASSERT_EQ(observer.Setup(config), Status::OK());
        ASSERT_TRUE(std::filesystem::exists(observer.DataFilePath(block, false)));
        auto lease = store->AcquireReadLease(&block, 1);
        EXPECT_EQ(bool(lease), early);
        if (early) { EXPECT_TRUE(observer.RemoveFile(block).Failure()); }
        Detail::TaskDesc load;
        load.push_back(Detail::Shard{block, 0, {target.data()}});
        auto loaded = store->Load(std::move(load));
        ASSERT_TRUE(loaded);
        EXPECT_EQ(store->Wait(loaded.Value()), Status::OK());
        EXPECT_EQ(source, target);
        store.reset();
        // A consumer's lease owns its fd, even after the creating Store is gone.
        if (early) {
            EXPECT_TRUE(observer.RemoveFile(block).Failure());
            lease.Value().reset();
        }
        EXPECT_EQ(observer.RemoveFile(block), Status::OK());
        EXPECT_FALSE(std::filesystem::exists(observer.DataFilePath(block, false)));
        EXPECT_EQ(std::filesystem::exists(path + "/.ucm_read_leases"), early);
    }
}

TEST_F(UCEarlyPosixStoreTest, FactoryPreservesNativeSetupAndLoadErrors)
{
    for (bool early : {false, true}) {
        std::unique_ptr<StoreV1> invalid{
            early ? MakeEarlyPosixStore() : MakePosixStore()};
        auto config = StoreConfig(Path());
        config.SetNumber("posix_lookup_concurrency", size_t{0});
        EXPECT_EQ(invalid->Setup(config), Status::InvalidParam());
        std::unique_ptr<StoreV1> store{
            early ? MakeEarlyPosixStore() : MakePosixStore()};
        ASSERT_EQ(store->Setup(StoreConfig(Path())), Status::OK());
        BlockId missing{};
        std::array<char, 4096> target{};
        Detail::TaskDesc load;
        load.push_back(Detail::Shard{missing, 0, {target.data()}});
        auto task = store->Load(std::move(load));
        ASSERT_TRUE(task);
        EXPECT_EQ(store->Wait(task.Value()), Status::NotFound());
    }
}
}  // namespace
