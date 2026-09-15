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
#include <filesystem>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fstream>
#include "detail/path_base.h"
#include "detail/types_helper.h"
#include "posix/cc/gc_lease.h"
#include "posix/cc/posix_file.h"
#include "posix/cc/space_manager.h"
#include "type/random_block_id.h"

class UCPosixSpaceManagerTest : public UC::Test::Detail::PathBase {};

TEST(RandomBlockIdTest, GeneratesDistinctIds)
{
    EXPECT_NE(UC::Detail::RandomBlockId(), UC::Detail::RandomBlockId());
}

TEST_F(UCPosixSpaceManagerTest, SetStorageBackends)
{
    using namespace UC::PosixStore;
    {
        SpaceManager spaceMgr;
        auto invalidPath = this->Path() + "invalid";
        Config config;
        config.storageBackends.push_back(std::move(invalidPath));
        auto s = spaceMgr.Setup(config);
        ASSERT_EQ(s, UC::Status::OsApiError());
    }
    {
        SpaceManager spaceMgr;
        auto validPath = this->Path();
        auto invalidPath = this->Path() + "invalid";
        Config config;
        config.storageBackends.push_back(std::move(validPath));
        config.storageBackends.push_back(std::move(invalidPath));
        auto s = spaceMgr.Setup(config);
        ASSERT_EQ(s, UC::Status::OK());
    }
    {
        SpaceManager spaceMgr;
        Config config;
        config.storageBackends.push_back(this->Path());
        config.storageBackends.push_back(this->Path());
        auto s = spaceMgr.Setup(config);
        ASSERT_EQ(s, UC::Status::OK());
    }
}

TEST_F(UCPosixSpaceManagerTest, RoundRobinUsesSharedBackendPaths)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    const auto mount0 = std::filesystem::path{Path()} / "mount0";
    const auto mount1 = std::filesystem::path{Path()} / "mount1";
    std::filesystem::create_directory(shared);
    std::filesystem::create_directory_symlink(shared, mount0);
    std::filesystem::create_directory_symlink(shared, mount1);
    Config config;
    config.dataDirShardBytes = 0;
    config.ioDirect = false;
    config.storageBackends = {mount0.string(), mount1.string()};
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), UC::Status::OK());
    const auto block = UC::Detail::RandomBlockId();
    for (size_t i = 0; i < 8; ++i) {
        auto path = layout.DataFilePath(block, true);
        ASSERT_TRUE(path.HasValue());
        const auto& mount = i % 2 == 0 ? mount0 : mount1;
        EXPECT_EQ(path.Value().find(mount.string() + "/"), 0);
    }
    auto temporary = layout.DataFilePath(block, true);
    ASSERT_TRUE(temporary.HasValue());
    std::ofstream(temporary.Value()) << "shared KV bytes";
    ASSERT_EQ(layout.CommitFile(block, true), UC::Status::OK());
    for (size_t i = 0; i < 2; ++i) {
        auto archived = layout.DataFilePath(block, false);
        ASSERT_TRUE(archived.HasValue());
        std::ifstream input(archived.Value());
        std::string content;
        std::getline(input, content);
        EXPECT_EQ(content, "shared KV bytes");
    }
}

TEST_F(UCPosixSpaceManagerTest, BackendMonitorsUseHealthThreadNames)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    const auto mount = std::filesystem::path{Path()} / "mount";
    std::filesystem::create_directory(shared);
    std::filesystem::create_directory_symlink(shared, mount);
    Config config;
    config.dataDirShardBytes = 0;
    config.storageBackends = {shared.string(), mount.string()};
    auto countMonitors = [] {
        size_t count = 0;
        for (const auto& entry : std::filesystem::directory_iterator("/proc/self/task")) {
            std::ifstream comm(entry.path() / "comm");
            std::string name;
            std::getline(comm, name);
            if (name == "ucm_health_pmon") { ++count; }
        }
        return count;
    };
    const auto before = countMonitors();
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), UC::Status::OK());
    auto count = before;
    for (size_t i = 0; i < 100 && count != before + 2; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        count = countMonitors();
    }
    EXPECT_EQ(count, before + 2);
}

TEST_F(UCPosixSpaceManagerTest, BackendProbesRemoveAndRestoreRoutes)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    const auto mount0 = std::filesystem::path{Path()} / "mount0";
    const auto mount1 = std::filesystem::path{Path()} / "mount1";
    std::filesystem::create_directory(shared);
    std::filesystem::create_directory_symlink(shared, mount0);
    std::filesystem::create_directory_symlink(shared, mount1);
    Config config;
    config.dataDirShardBytes = 0;
    config.ioDirect = false;
    config.storageBackends = {mount0.string(), mount1.string()};
    config.backendHealth.healthCheckInterval = std::chrono::milliseconds(50);
    config.backendHealth.healthCheckTimeout = std::chrono::milliseconds(40);
    config.backendHealth.healthWindowSize = 4;
    config.backendHealth.failureThreshold = 2;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), UC::Status::OK());
    const auto block = UC::Detail::RandomBlockId();
    auto waitUntil = [](auto condition) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!condition()) {
            if (std::chrono::steady_clock::now() >= deadline) { return false; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return true;
    };
    const auto onlyRoute = [&](const std::filesystem::path& mount) {
        for (size_t i = 0; i < 4; ++i) {
            auto path = layout.DataFilePath(block, false);
            if (!path || path.Value().find(mount.string() + "/") != 0) { return false; }
        }
        return true;
    };

    std::filesystem::remove(mount0);
    ASSERT_TRUE(waitUntil([&] { return onlyRoute(mount1); }));
    EXPECT_EQ(layout.CheckHealth(), UC::Status::OK());
    std::filesystem::remove(mount1);
    ASSERT_TRUE(waitUntil([&] { return layout.CheckHealth() == UC::Status::StoreUnhealthy(); }));
    EXPECT_EQ(layout.DataFilePath(block, false).Error(), UC::Status::StoreUnhealthy());
    EXPECT_EQ(layout.CommitFile(block, true), UC::Status::StoreUnhealthy());
    EXPECT_EQ(layout.RemoveFile(block), UC::Status::StoreUnhealthy());

    std::filesystem::create_directory_symlink(shared, mount0);
    ASSERT_TRUE(waitUntil([&] { return layout.CheckHealth().Success(); }));
    EXPECT_TRUE(onlyRoute(mount0));
    std::filesystem::create_directory_symlink(shared, mount1);
    ASSERT_TRUE(waitUntil([&] { return !onlyRoute(mount0); }));
    auto first = layout.DataFilePath(block, false);
    auto second = layout.DataFilePath(block, false);
    ASSERT_TRUE(first.HasValue());
    ASSERT_TRUE(second.HasValue());
    EXPECT_NE(first.Value(), second.Value());
}

TEST_F(UCPosixSpaceManagerTest, StartsWithAnUnavailableBackendAndRecoversIt)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    const auto mount = std::filesystem::path{Path()} / "missing";
    std::filesystem::create_directory(shared);
    Config config;
    config.dataDirShardBytes = 0;
    config.ioDirect = false;
    config.storageBackends = {mount.string(), shared.string()};
    config.backendHealth.healthCheckInterval = std::chrono::milliseconds(50);
    config.backendHealth.healthCheckTimeout = std::chrono::milliseconds(40);
    config.backendHealth.healthWindowSize = 4;
    config.backendHealth.failureThreshold = 2;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), UC::Status::OK());
    const auto block = UC::Detail::RandomBlockId();
    ASSERT_EQ(layout.CheckHealth(), UC::Status::OK());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(layout.DataFilePath(block, false).Value().find(shared.string()), 0);
    }
    std::filesystem::create_directory_symlink(shared, mount);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool recovered = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (layout.DataFilePath(block, false).Value().find(mount.string()) == 0) {
            recovered = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_TRUE(recovered);
}

TEST_F(UCPosixSpaceManagerTest, GcLeaseSurvivesBackendFailover)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    const auto mount0 = std::filesystem::path{Path()} / "mount0";
    const auto mount1 = std::filesystem::path{Path()} / "mount1";
    std::filesystem::create_directory(shared);
    std::filesystem::create_directory_symlink(shared, mount1);
    Config config;
    config.storageBackends = {mount0.string(), mount1.string()};
    config.dataDirShardBytes = 0;
    config.ioDirect = false;
    config.backendHealth.healthCheckInterval = std::chrono::milliseconds(50);
    config.backendHealth.healthCheckTimeout = std::chrono::milliseconds(40);
    config.backendHealth.healthWindowSize = 4;
    config.backendHealth.failureThreshold = 2;
    config.posixGcHeartbeatIntervalSec = 1;
    config.posixGcStaleThresholdSec = 10;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), UC::Status::OK());
    GcLease holder, peer;
    holder.Setup(config, &layout);
    peer.Setup(config, &layout);
    ASSERT_EQ(holder.TryAcquire(), GcLease::Acquisition::Acquired);
    ASSERT_EQ(peer.TryAcquire(), GcLease::Acquisition::HeldByPeer);

    auto waitUntil = [](auto condition) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            if (condition()) { return true; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
    };
    std::filesystem::create_directory_symlink(shared, mount0);
    ASSERT_TRUE(waitUntil(
        [&] { return layout.StorageBackend().Value() != layout.StorageBackend().Value(); }));
    std::filesystem::remove(mount1);
    ASSERT_TRUE(waitUntil([&] {
        return layout.StorageBackend().Value() == mount0.string() + "/" &&
               layout.StorageBackend().Value() == mount0.string() + "/";
    }));
    const auto lockDir = shared / ".ucm_gc.lock";
    const auto heartbeat = std::filesystem::directory_iterator(lockDir)->path();
    const auto oldStamp = std::filesystem::file_time_type::clock::now() - std::chrono::seconds(5);
    std::filesystem::last_write_time(heartbeat, oldStamp);
    ASSERT_TRUE(waitUntil([&] { return std::filesystem::last_write_time(heartbeat) > oldStamp; }));
    EXPECT_TRUE(holder.HoldsLock());
    EXPECT_EQ(peer.TryAcquire(), GcLease::Acquisition::HeldByPeer);
    holder.Release();
    EXPECT_FALSE(std::filesystem::exists(lockDir));
    EXPECT_EQ(peer.TryAcquire(), GcLease::Acquisition::Acquired);
    EXPECT_TRUE(peer.HoldsLock());
    peer.Release();
    EXPECT_FALSE(std::filesystem::exists(lockDir));
}

TEST_F(UCPosixSpaceManagerTest, DataFilePath)
{
    using namespace UC::PosixStore;
    SpaceManager spaceMgr;
    Config config;
    config.dataDirShardBytes = 0;
    config.storageBackends.push_back(this->Path());
    auto s = spaceMgr.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    auto activated = spaceMgr.GetLayout()->DataFilePath(blockId, true).Value();
    ASSERT_EQ(activated, fmt::format("{}data/{:02x}.tmp", this->Path(), fmt::join(blockId, "")));
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    ASSERT_EQ(PosixFile{activated}.Open(PosixFile::OpenFlag::CREATE), UC::Status::OK());
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::OK());
    ASSERT_EQ(spaceMgr.Lookup(&blockId, 1).Value(), std::vector<uint8_t>{false});
    ASSERT_EQ(spaceMgr.GetLayout()->CommitFile(blockId, true), UC::Status::OK());
    ASSERT_EQ(spaceMgr.Lookup(&blockId, 1).Value(), std::vector<uint8_t>{true});
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    auto archived = spaceMgr.GetLayout()->DataFilePath(blockId, false).Value();
    ASSERT_EQ(archived, fmt::format("{}data/{:02x}", this->Path(), fmt::join(blockId, "")));
    ASSERT_EQ(PosixFile{archived}.Access(PosixFile::AccessMode::EXIST), UC::Status::OK());
}

TEST_F(UCPosixSpaceManagerTest, ShardFilePath)
{
    using namespace UC::PosixStore;
    SpaceManager spaceMgr;
    Config config;
    config.dataDirShardBytes = 2;
    config.storageBackends.push_back(this->Path());
    auto s = spaceMgr.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    const auto& file = fmt::format("{:02x}", fmt::join(blockId, ""));
    const auto& shard = file.substr(0, config.dataDirShardBytes);
    auto activated = spaceMgr.GetLayout()->DataFilePath(blockId, true).Value();
    ASSERT_EQ(activated, fmt::format("{}{}/{}.tmp", this->Path(), shard, file));
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    ASSERT_EQ(PosixFile{activated}.Open(PosixFile::OpenFlag::CREATE), UC::Status::OK());
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::OK());
    ASSERT_EQ(spaceMgr.Lookup(&blockId, 1).Value(), std::vector<uint8_t>{false});
    ASSERT_EQ(spaceMgr.GetLayout()->CommitFile(blockId, true), UC::Status::OK());
    ASSERT_EQ(spaceMgr.Lookup(&blockId, 1).Value(), std::vector<uint8_t>{true});
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    auto archived = spaceMgr.GetLayout()->DataFilePath(blockId, false).Value();
    ASSERT_EQ(archived, fmt::format("{}{}/{}", this->Path(), shard, file));
    ASSERT_EQ(PosixFile{archived}.Access(PosixFile::AccessMode::EXIST), UC::Status::OK());
}

TEST_F(UCPosixSpaceManagerTest, Lookup)
{
    using namespace UC::PosixStore;
    SpaceManager spaceMgr;
    Config config;
    config.dataDirShardBytes = 0;
    config.storageBackends.push_back(Path());
    ASSERT_TRUE(spaceMgr.Setup(config).Success());
    std::vector<UC::Detail::BlockId> blocks(3);
    std::for_each(blocks.begin(), blocks.end(), [](auto& block) {
        block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    });
    {
        auto foundIdx = spaceMgr.LookupOnPrefix(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(foundIdx, -1);
        auto ReverseIdx = spaceMgr.LookupOnReverse(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(ReverseIdx, -1);
        auto founds = spaceMgr.Lookup(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(founds.size(), blocks.size());
        std::for_each(founds.begin(), founds.end(), [](auto found) { ASSERT_FALSE(found); });
    }
    std::for_each(blocks.begin(), blocks.end(), [&](const auto& block) {
        auto archived = spaceMgr.GetLayout()->DataFilePath(block, false).Value();
        ASSERT_EQ(PosixFile{archived}.Open(PosixFile::OpenFlag::CREATE), UC::Status::OK());
    });
    {
        auto foundIdx = spaceMgr.LookupOnPrefix(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(foundIdx, 2);
        auto ReverseIdx = spaceMgr.LookupOnReverse(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(ReverseIdx, 2);
        auto founds = spaceMgr.Lookup(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(founds.size(), blocks.size());
        std::for_each(founds.begin(), founds.end(), [](auto found) { ASSERT_TRUE(found); });
    }
    auto pos = blocks.begin();
    std::advance(pos, 2);
    blocks.insert(pos, UC::Test::Detail::TypesHelper::MakeBlockIdRandomly());
    {
        auto foundIdx = spaceMgr.LookupOnPrefix(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(foundIdx, 1);
        auto ReverseIdx = spaceMgr.LookupOnReverse(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(ReverseIdx, 3);
        auto founds = spaceMgr.Lookup(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(founds.size(), blocks.size());
        std::vector<uint8_t> expected{true, true, false, false};
        ASSERT_TRUE(founds == expected);
    }
}
