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
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fstream>
#include <set>
#include <sys/syscall.h>
#include "detail/path_base.h"
#include "detail/types_helper.h"
#include "posix/cc/gc_lease.h"
#include "posix/cc/posix_file.h"
#include "posix/cc/space_manager.h"
#include "type/random_block_id.h"

class UCPosixSpaceManagerTest : public UC::Test::Detail::PathBase {};

static UC::Detail::BlockId BlockForBackend(size_t backend, size_t count)
{
    UC::Detail::BlockId block{};
    for (size_t value = 0; value < 256; ++value) {
        block[0] = static_cast<std::byte>(value);
        if (UC::Detail::BlockIdHasher{}(block) % count == backend) { return block; }
    }
    ADD_FAILURE() << "No test block mapped to backend " << backend;
    return block;
}

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
        config.storageBackends.push_back(invalidPath);
        auto s = spaceMgr.Setup(config);
        ASSERT_EQ(s, UC::Status::NotFound());
        EXPECT_NE(s.ToString().find(invalidPath), std::string::npos);
        EXPECT_NE(s.ToString().find("access("), std::string::npos);
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

TEST_F(UCPosixSpaceManagerTest, BlockHashUsesSharedBackendPaths)
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
    const auto block = BlockForBackend(0, 2);
    const auto other = BlockForBackend(1, 2);
    for (size_t i = 0; i < 8; ++i) {
        auto path = layout.DataFilePath(block, true);
        ASSERT_TRUE(path.HasValue());
        EXPECT_EQ(path.Value().find(mount0.string() + "/"), 0);
        EXPECT_EQ(layout.StorageBackend(other).Value(), mount1.string() + "/");
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

TEST_F(UCPosixSpaceManagerTest, HashRoutingKeepsTotalBackendCountAndWalksForward)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    std::filesystem::create_directory(shared);
    Config config;
    config.dataDirShardBytes = 0;
    config.backendHealth.healthCheckInterval = std::chrono::hours(1);
    config.backendHealth.failureThreshold = 1;
    std::array<UC::Detail::BlockId, 4> blocks;
    for (size_t i = 0; i < blocks.size(); ++i) {
        const auto mount = std::filesystem::path{Path()} / ("mount" + std::to_string(i));
        std::filesystem::create_directory_symlink(shared, mount);
        config.storageBackends.push_back(mount.string());
        blocks[i] = BlockForBackend(i, blocks.size());
    }
    SpaceLayout layout, peer;
    ASSERT_EQ(layout.Setup(config), UC::Status::OK());
    ASSERT_EQ(peer.Setup(config), UC::Status::OK());
    const auto& backends = layout.Backends();
    for (size_t i = 0; i < blocks.size(); ++i) {
        EXPECT_EQ(layout.StorageBackend(blocks[i]).Value(), backends[i]);
        EXPECT_EQ(peer.StorageBackend(blocks[i]).Value(), backends[i]);
    }
    std::vector<std::string> attempts;
    EXPECT_EQ(layout.RunOnAvailableBackend(blocks[3],
                                           [&](const std::string& backend) {
                                               attempts.push_back(backend);
                                               return backend == backends[1]
                                                          ? UC::Status::OK()
                                                          : UC::Status::OsApiError();
                                           }),
              UC::Status::OK());
    EXPECT_EQ(attempts, (std::vector<std::string>{backends[3], backends[0], backends[1]}));
    EXPECT_EQ(layout.BackendCount(), 4);
    EXPECT_EQ(layout.StorageBackend(blocks[0]).Value(), backends[1]);
    EXPECT_EQ(layout.StorageBackend(blocks[1]).Value(), backends[1]);
    EXPECT_EQ(layout.StorageBackend(blocks[2]).Value(), backends[2]);
    EXPECT_EQ(layout.StorageBackend(blocks[3]).Value(), backends[1]);
    EXPECT_EQ(layout.StorageBackend(blocks[3], {backends[1]}).Value(), backends[2]);
    EXPECT_EQ(layout.StorageBackend(blocks[3], {backends[1], backends[2]}).Error(),
              UC::Status::StoreUnhealthy());
    layout.RecordIoResult(backends[1], UC::Status::Timeout());
    layout.RecordIoResult(backends[2], UC::Status::Timeout());
    EXPECT_EQ(layout.StorageBackend(blocks[3]).Error(), UC::Status::StoreUnhealthy());
}

TEST_F(UCPosixSpaceManagerTest, OnlyOneRecoveryMonitorUsesHealthThreadName)
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
    for (size_t i = 0; i < 100 && count != before + 1; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        count = countMonitors();
    }
    EXPECT_EQ(count, before + 1);
}

TEST_F(UCPosixSpaceManagerTest, PassiveFailuresRemoveRoutesAndProbesRestoreThem)
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
    const auto block = BlockForBackend(0, 2);
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
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    EXPECT_FALSE(onlyRoute(mount1));
    layout.RecordIoResult(mount0.string() + "/", UC::Status::OsApiError());
    layout.RecordIoResult(mount0.string() + "/", UC::Status::OsApiError());
    ASSERT_TRUE(onlyRoute(mount1));
    EXPECT_EQ(layout.CheckHealth(), UC::Status::OK());
    std::filesystem::remove(mount1);
    layout.RecordIoResult(mount1.string() + "/", UC::Status::Timeout());
    layout.RecordIoResult(mount1.string() + "/", UC::Status::Timeout());
    ASSERT_EQ(layout.CheckHealth(), UC::Status::StoreUnhealthy());
    EXPECT_EQ(layout.DataFilePath(block, false).Error(), UC::Status::StoreUnhealthy());
    EXPECT_EQ(layout.CommitFile(block, true), UC::Status::StoreUnhealthy());
    EXPECT_EQ(layout.RemoveFile(block), UC::Status::StoreUnhealthy());

    std::filesystem::create_directory_symlink(shared, mount0);
    ASSERT_TRUE(waitUntil([&] { return layout.CheckHealth().Success(); }));
    EXPECT_TRUE(onlyRoute(mount0));
    std::filesystem::create_directory_symlink(shared, mount1);
    const auto other = BlockForBackend(1, 2);
    ASSERT_TRUE(
        waitUntil([&] { return layout.StorageBackend(other).Value() == mount1.string() + "/"; }));
    EXPECT_EQ(layout.StorageBackend(block).Value(), mount0.string() + "/");
    EXPECT_EQ(layout.DataFilePath(block, false).Value(), layout.DataFilePath(block, false).Value());
}

TEST_F(UCPosixSpaceManagerTest, RejectsAnUnavailableBackendInEitherPosition)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    const auto missing = std::filesystem::path{Path()} / "missing";
    std::filesystem::create_directory(shared);
    for (const auto missingFirst : {true, false}) {
        Config config;
        config.dataDirShardBytes = 0;
        config.ioDirect = false;
        config.storageBackends = missingFirst
                                     ? std::vector<std::string>{missing.string(), shared.string()}
                                     : std::vector<std::string>{shared.string(), missing.string()};
        SpaceLayout layout;
        const auto status = layout.Setup(config);
        EXPECT_TRUE(status.Failure());
        EXPECT_NE(status.ToString().find(missing.string()), std::string::npos);
        EXPECT_NE(status.ToString().find(missingFirst ? "mkdir(" : "access("), std::string::npos);
        EXPECT_NE(status.ToString().find("errno=" + std::to_string(ENOENT)), std::string::npos);
        EXPECT_NE(status.ToString().find(std::strerror(ENOENT)), std::string::npos);
        EXPECT_EQ(layout.CheckHealth(), UC::Status::StoreUnhealthy());
    }
}

TEST_F(UCPosixSpaceManagerTest, StartupProbesEveryBackendAndCleansUp)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    const auto mount = std::filesystem::path{Path()} / "mount";
    std::filesystem::create_directory(shared);
    std::filesystem::create_directory_symlink(shared, mount);
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    for (const auto direct : {false, true}) {
        Config config;
        config.dataDirShardBytes = 0;
        config.ioDirect = direct;
        config.backendHealth.enabled = false;
        config.storageBackends = {shared.string(), mount.string(), mount.string() + "/"};
        std::vector<std::string> paths;
        TestHooks::SetOpenHook([&](const std::string& path, int32_t flags, mode_t mode) {
            paths.push_back(path);
            EXPECT_EQ(flags & O_ACCMODE, O_RDWR);
            EXPECT_EQ((flags & O_DIRECT) != 0, direct);
            return ::open(path.c_str(), flags, mode);
        });
        SpaceLayout layout;
        ASSERT_EQ(layout.Setup(config), UC::Status::OK());
        ASSERT_EQ(paths.size(), 2);
        EXPECT_EQ(paths[0].find(shared.string() + "/"), 0);
        EXPECT_EQ(paths[1].find(mount.string() + "/"), 0);
        EXPECT_EQ(layout.BackendCount(), 2);
        EXPECT_EQ(layout.CheckHealth(), UC::Status::OK());
        EXPECT_TRUE(std::filesystem::is_empty(shared / "data"));
        TestHooks::ClearOpenHook();
    }
}

TEST_F(UCPosixSpaceManagerTest, GcLeaseSurvivesBackendFailover)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    const auto mount0 = std::filesystem::path{Path()} / "mount0";
    const auto mount1 = std::filesystem::path{Path()} / "mount1";
    std::filesystem::create_directory(shared);
    std::filesystem::create_directory_symlink(shared, mount0);
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
    std::filesystem::remove(mount0);
    layout.RecordIoResult(mount0.string() + "/", UC::Status::OsApiError());
    layout.RecordIoResult(mount0.string() + "/", UC::Status::OsApiError());
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
    ASSERT_TRUE(waitUntil([&] {
        return layout.StorageBackend(BlockForBackend(0, 2)).Value() == mount0.string() + "/";
    }));
    std::filesystem::remove(mount1);
    layout.RecordIoResult(mount1.string() + "/", UC::Status::OsApiError());
    layout.RecordIoResult(mount1.string() + "/", UC::Status::OsApiError());
    ASSERT_TRUE(waitUntil([&] {
        return layout.StorageBackend(BlockForBackend(1, 2)).Value() == mount0.string() + "/";
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

TEST_F(UCPosixSpaceManagerTest, UsesTotalBackendCountForTimeoutAndIgnoresMissingFiles)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    const auto mount = std::filesystem::path{Path()} / "mount";
    const auto mount2 = std::filesystem::path{Path()} / "mount2";
    std::filesystem::create_directory(shared);
    std::filesystem::create_directory_symlink(shared, mount);
    std::filesystem::create_directory_symlink(shared, mount2);
    Config config;
    config.dataDirShardBytes = 0;
    config.timeoutMs = 900;
    config.storageBackends = {shared.string(), mount.string(), mount2.string()};
    config.backendHealth.healthCheckInterval = std::chrono::hours(1);
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), UC::Status::OK());
    std::filesystem::remove(mount2);
    layout.RecordIoResult(mount2.string() + "/", UC::Status::OsApiError());
    layout.RecordIoResult(mount2.string() + "/", UC::Status::OsApiError());
    EXPECT_EQ(layout.BackendCount(), 3);
    EXPECT_EQ(layout.IoTimeoutMs(), 300);
    for (size_t i = 0; i < 20; ++i) {
        layout.RecordIoResult(shared.string() + "/", UC::Status::NotFound());
    }
    EXPECT_TRUE(layout.StorageBackend({}, {mount.string() + "/"}).HasValue());
    layout.RecordIoResult(shared.string() + "/", UC::Status::OsApiError());
    for (size_t i = 0; i < 8; ++i) {
        layout.RecordIoResult(shared.string() + "/", UC::Status::OK());
    }
    layout.RecordIoResult(shared.string() + "/", UC::Status::OsApiError());
    EXPECT_TRUE(layout.StorageBackend({}, {mount.string() + "/"}).HasValue());
    layout.RecordIoResult(shared.string() + "/", UC::Status::OsApiError());
    EXPECT_FALSE(layout.StorageBackend({}, {mount.string() + "/"}).HasValue());
    EXPECT_EQ(layout.IoTimeoutMs(), 300);
}

TEST_F(UCPosixSpaceManagerTest, NonTransportErrorsDoNotAdvanceTheHealthWindow)
{
    using namespace UC::PosixStore;
    Config config;
    config.dataDirShardBytes = 0;
    config.ioDirect = false;
    config.storageBackends = {Path()};
    config.backendHealth.healthCheckInterval = std::chrono::hours(1);
    config.backendHealth.healthWindowSize = 3;
    config.backendHealth.failureThreshold = 2;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), UC::Status::OK());
    const auto& backend = layout.Backends().front();
    layout.RecordIoResult(backend, UC::Status::OsApiError(EIO));
    for (const int error :
         {ENOENT, EEXIST,    ENOTDIR, EISDIR, ENOTEMPTY, ELOOP,      ENAMETOOLONG, ESTALE,
          EACCES, EPERM,     EROFS,   ENOSPC, EDQUOT,    EFBIG,      EOVERFLOW,    EMLINK,
          EXDEV,  EINVAL,    EBADF,   EFAULT, ESPIPE,    EOPNOTSUPP, ENOSYS,       EAGAIN,
          EINTR,  ECANCELED, ENOMEM,  EMFILE, ENFILE,    EBUSY,      ETXTBSY}) {
        SCOPED_TRACE(error);
        layout.RecordIoResult(backend, UC::Status::OsApiError(error));
        layout.RecordIoResult(backend, UC::Status::OsApiError(error));
        EXPECT_EQ(layout.CheckHealth(), UC::Status::OK());
    }
    for (const auto& status : {
             UC::Status::NotFound(), UC::Status::InvalidParam(), UC::Status::OutOfMemory(),
             UC::Status::DuplicateKey(), UC::Status::Retry(), UC::Status::Unsupported(),
             UC::Status::NoSpace(),
             UC::Status{UC::Status::Timeout().Underlying(), "queue full", EAGAIN}
    }) {
        layout.RecordIoResult(backend, status);
        EXPECT_EQ(layout.CheckHealth(), UC::Status::OK());
    }
    layout.RecordIoResult(backend, UC::Status::OsApiError(ECONNRESET));
    EXPECT_EQ(layout.CheckHealth(), UC::Status::StoreUnhealthy());
}

TEST_F(UCPosixSpaceManagerTest, OnlyExcludedBackendsReceiveRecoveryIo)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    const auto mount = std::filesystem::path{Path()} / "mount";
    std::filesystem::create_directory(shared);
    std::filesystem::create_directory_symlink(shared, mount);
    Config config;
    config.dataDirShardBytes = 0;
    config.ioDirect = false;
    config.storageBackends = {shared.string(), mount.string()};
    config.backendHealth.healthCheckInterval = std::chrono::milliseconds(30);
    config.backendHealth.healthCheckTimeout = std::chrono::milliseconds(20);
    config.backendHealth.healthWindowSize = 4;
    config.backendHealth.failureThreshold = 2;
    auto probeCount = std::make_shared<std::atomic<size_t>>(0);
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), UC::Status::OK());
    TestHooks::SetOpenHook(
        [probeCount, shared](const std::string& path, int32_t flags, mode_t mode) {
            EXPECT_EQ(path.find(shared.string() + "/"), 0);
            ++*probeCount;
            return ::open(path.c_str(), flags, mode);
        });
    for (size_t i = 0; i < 4; ++i) { EXPECT_TRUE(layout.CheckHealth().Success()); }
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    EXPECT_EQ(probeCount->load(), 0);
    layout.RecordIoResult(shared.string() + "/", UC::Status::OsApiError());
    layout.RecordIoResult(shared.string() + "/", UC::Status::OsApiError());
    for (size_t i = 0; i < 20; ++i) {
        layout.RecordIoResult(shared.string() + "/", UC::Status::OK());
    }
    EXPECT_FALSE(layout.StorageBackend({}, {mount.string() + "/"}).HasValue());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!layout.StorageBackend({}, {mount.string() + "/"}).HasValue() &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_TRUE(layout.StorageBackend({}, {mount.string() + "/"}).HasValue());
    EXPECT_EQ(probeCount->load(), 4);
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    EXPECT_EQ(probeCount->load(), 4);
}

TEST_F(UCPosixSpaceManagerTest, StalledRecoveryDoesNotBlockOtherBackends)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    const auto mount = std::filesystem::path{Path()} / "mount";
    std::filesystem::create_directory(shared);
    std::filesystem::create_directory_symlink(shared, mount);
    Config config;
    config.dataDirShardBytes = 0;
    config.ioDirect = false;
    config.storageBackends = {shared.string(), mount.string()};
    config.backendHealth.healthCheckInterval = std::chrono::milliseconds(30);
    config.backendHealth.healthCheckTimeout = std::chrono::milliseconds(20);
    config.backendHealth.healthWindowSize = 4;
    config.backendHealth.failureThreshold = 1;
    struct ProbeState {
        std::mutex mutex;
        std::condition_variable cv;
        bool release{false};
        std::atomic<size_t> stalledCalls{0};
    };
    auto state = std::make_shared<ProbeState>();
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), UC::Status::OK());
    struct ReleaseProbe {
        std::shared_ptr<ProbeState> state;
        ~ReleaseProbe()
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->release = true;
            state->cv.notify_all();
        }
    } release{state};
    TestHooks::SetOpenHook([state, shared](const std::string& path, int32_t flags, mode_t mode) {
        if (path.find(shared.string() + "/") == 0) {
            ++state->stalledCalls;
            std::unique_lock<std::mutex> lock(state->mutex);
            state->cv.wait(lock, [&] { return state->release; });
        }
        return ::open(path.c_str(), flags, mode);
    });
    layout.RecordIoResult(shared.string() + "/", UC::Status::Timeout());
    layout.RecordIoResult(mount.string() + "/", UC::Status::Timeout());
    EXPECT_EQ(layout.CheckHealth(), UC::Status::StoreUnhealthy());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (layout.CheckHealth().Failure() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_TRUE(layout.CheckHealth().Success());
    EXPECT_EQ(layout.StorageBackend({}).Value(), mount.string() + "/");
    EXPECT_GE(state->stalledCalls.load(), 2);
}

TEST_F(UCPosixSpaceManagerTest, RecoveryReusesOneWorkerAcrossBackends)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    const auto mount = std::filesystem::path{Path()} / "mount";
    std::filesystem::create_directory(shared);
    std::filesystem::create_directory_symlink(shared, mount);
    Config config;
    config.dataDirShardBytes = 0;
    config.ioDirect = false;
    config.storageBackends = {shared.string(), mount.string()};
    config.backendHealth.healthCheckInterval = std::chrono::milliseconds(80);
    config.backendHealth.healthCheckTimeout = std::chrono::milliseconds(50);
    config.backendHealth.healthWindowSize = 4;
    config.backendHealth.failureThreshold = 1;
    std::mutex mutex;
    std::set<pid_t> workers;
    std::vector<std::string> probes;
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), UC::Status::OK());
    TestHooks::SetOpenHook([&](const std::string& path, int32_t flags, mode_t mode) {
        std::lock_guard<std::mutex> lock(mutex);
        workers.insert(static_cast<pid_t>(syscall(SYS_gettid)));
        probes.push_back(path);
        return ::open(path.c_str(), flags, mode);
    });
    for (const auto& backend : layout.Backends()) {
        layout.RecordIoResult(backend, UC::Status::OsApiError());
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((!layout.StorageBackend({}, {mount.string() + "/"}) ||
            !layout.StorageBackend({}, {shared.string() + "/"})) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_TRUE(layout.StorageBackend({}, {mount.string() + "/"}).HasValue());
    EXPECT_TRUE(layout.StorageBackend({}, {shared.string() + "/"}).HasValue());
    std::lock_guard<std::mutex> lock(mutex);
    EXPECT_EQ(workers.size(), 1);
    EXPECT_EQ(probes.size(), 8);
    for (size_t i = 0; i < probes.size(); ++i) {
        const auto& backend = layout.Backends()[i % 2];
        EXPECT_EQ(probes[i].find(backend), 0);
    }
}

TEST_F(UCPosixSpaceManagerTest, BackendRecoversWhileItsOldProbeIsStillBlocked)
{
    using namespace UC::PosixStore;
    Config config;
    config.dataDirShardBytes = 0;
    config.ioDirect = false;
    config.storageBackends = {Path()};
    config.backendHealth.healthCheckInterval = std::chrono::milliseconds(80);
    config.backendHealth.healthCheckTimeout = std::chrono::milliseconds(50);
    config.backendHealth.healthWindowSize = 4;
    config.backendHealth.failureThreshold = 1;
    struct ProbeState {
        std::mutex mutex;
        std::condition_variable cv;
        bool release{false};
        size_t calls{0};
        std::set<pid_t> workers;
    };
    auto state = std::make_shared<ProbeState>();
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), UC::Status::OK());
    struct ReleaseProbe {
        std::shared_ptr<ProbeState> state;
        ~ReleaseProbe()
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->release = true;
            state->cv.notify_all();
        }
    } release{state};
    TestHooks::SetOpenHook([state](const std::string& path, int32_t flags, mode_t mode) {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->workers.insert(static_cast<pid_t>(syscall(SYS_gettid)));
        if (++state->calls == 1) {
            state->cv.wait(lock, [&] { return state->release; });
        }
        return ::open(path.c_str(), flags, mode);
    });
    layout.RecordIoResult(layout.Backends().front(), UC::Status::OsApiError());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (layout.CheckHealth().Failure() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(layout.CheckHealth(), UC::Status::OK());
    std::lock_guard<std::mutex> lock(state->mutex);
    EXPECT_FALSE(state->release);
    EXPECT_EQ(state->calls, 5);
    EXPECT_EQ(state->workers.size(), 2);
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
