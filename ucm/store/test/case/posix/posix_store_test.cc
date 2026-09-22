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
#include "posix/cc/posix_store.cc"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <sys/syscall.h>
#include <thread>
#include <vector>
#include "detail/data_generator.h"
#include "detail/path_base.h"
#include "detail/types_helper.h"
#include "metrics_api.h"
#include "posix/cc/aio_impl.h"

class UCPosixStoreTest : public UC::Test::Detail::PathBase {};

namespace {
constexpr size_t AIO_TEST_DATA_SIZE = 4096;
using BufferPtr = std::unique_ptr<void, decltype(&std::free)>;

UC::Detail::Dictionary MakeAioConfig(const std::string& path, size_t timeoutMs = 50,
                                     size_t openConcurrency = 1, size_t shardsPerBlock = 1)
{
    UC::Detail::Dictionary config;
    config.SetNumber("device_id", 0);
    config.Set("storage_backends", std::vector<std::string>{path});
    config.SetNumber("tensor_size", AIO_TEST_DATA_SIZE);
    config.SetNumber("shard_size", AIO_TEST_DATA_SIZE);
    config.SetNumber("block_size", AIO_TEST_DATA_SIZE * shardsPerBlock);
    config.Set("posix_io_engine", std::string("aio"));
    config.SetNumber("timeout_ms", timeoutMs);
    config.SetNumber("posix_open_concurrency", openConcurrency);
    config.SetNumber("posix_commit_concurrency", size_t(1));
    config.SetNumber("data_dir_shard_bytes", size_t(0));
    return config;
}

UC::Detail::Dictionary MakePsyncConfig(const std::string& path)
{
    UC::Detail::Dictionary config;
    config.SetNumber("device_id", 0);
    config.Set("storage_backends", std::vector<std::string>{path});
    config.SetNumber("tensor_size", AIO_TEST_DATA_SIZE);
    config.SetNumber("shard_size", AIO_TEST_DATA_SIZE);
    config.SetNumber("block_size", AIO_TEST_DATA_SIZE);
    config.Set("posix_io_engine", std::string("psync"));
    config.Set("io_direct", false);
    config.SetNumber("data_dir_shard_bytes", size_t(0));
    return config;
}

BufferPtr MakeAlignedBuffer(size_t marker)
{
    void* buffer = nullptr;
    if (posix_memalign(&buffer, 4096, AIO_TEST_DATA_SIZE) != 0) { return {nullptr, &std::free}; }
    std::memset(buffer, 0, AIO_TEST_DATA_SIZE);
    *reinterpret_cast<size_t*>(buffer) = marker;
    return {buffer, &std::free};
}

UC::Detail::TaskDesc MakeDumpDesc(const char* brief, const UC::Detail::BlockId& block, void* buffer)
{
    UC::Detail::TaskDesc desc;
    desc.brief = brief;
    desc.push_back(UC::Detail::Shard{block, 0, {buffer}});
    return desc;
}

void RegisterCounter(const std::string& name)
{
    UC::Metrics::SetUp();
    UC::Metrics::CreateStats(name, "counter");
    UC::Metrics::GetAllStatsAndClear();
}

double ReadCounter(const std::string& name)
{
    const auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    auto it = counters.find(name);
    return it == counters.end() ? 0.0 : it->second;
}

class StallingOpenHook {
public:
    explicit StallingOpenHook(bool succeedAfterRelease = false)
        : succeedAfterRelease_{succeedAfterRelease}
    {
        UC::PosixStore::TestHooks::SetOpenHook(
            [this](const std::string& path, int32_t flags, mode_t mode) {
                return Run(path, flags, mode);
            });
    }
    ~StallingOpenHook()
    {
        {
            std::lock_guard<std::mutex> lock{mutex_};
            release_ = true;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lock{mutex_};
        cv_.wait(lock, [this] { return active_ == 0; });
        UC::PosixStore::TestHooks::ClearOpenHook();
    }
    bool WaitEntered(size_t timeoutMs = 1000)
    {
        std::unique_lock<std::mutex> lock{mutex_};
        return cv_.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                            [this] { return entered_; });
    }

private:
    int32_t Run(const std::string& path, int32_t flags, mode_t mode)
    {
        {
            std::lock_guard<std::mutex> lock{mutex_};
            ++active_;
            entered_ = true;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lock{mutex_};
        cv_.wait(lock, [this] { return release_; });
        auto succeed = succeedAfterRelease_;
        lock.unlock();
        auto fd = succeed ? ::open(path.c_str(), flags, mode) : -1;
        auto err = succeed ? errno : EIO;
        lock.lock();
        --active_;
        cv_.notify_all();
        errno = err;
        return fd;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_{false};
    bool release_{false};
    bool succeedAfterRelease_{false};
    size_t active_{0};
};

class ScopedAioHooks {
public:
    ~ScopedAioHooks() { UC::PosixStore::TestHooks::ClearAioHooks(); }
};
}  // namespace

TEST_F(UCPosixStoreTest, SetupWithInvalidParam)
{
    using namespace UC::PosixStore;
    {
        UC::Detail::Dictionary config;
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::InvalidParam());
    }
    {
        UC::Detail::Dictionary config;
        config.Set("storage_backends", std::vector<std::string>{Path()});
        config.SetNumber("device_id", 0);
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::InvalidParam());
    }
    {
        UC::Detail::Dictionary config;
        config.Set("storage_backends", std::vector<std::string>{Path()});
        config.SetNumber("device_id", 0);
        config.SetNumber("tensor_size", size_t(4096));
        config.SetNumber("shard_size", size_t(4096));
        config.SetNumber("block_size", size_t(4096));
        config.Set("posix_io_engine", std::string("psync"));
        config.SetNumber("posix_data_trans_concurrency", size_t(0));
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::InvalidParam());
    }
}

TEST_F(UCPosixStoreTest, DumpThenLoad)
{
    using namespace UC::PosixStore;
    UC::Detail::Dictionary config;
    config.SetNumber("device_id", 0);
    config.Set("storage_backends", std::vector<std::string>{Path()});
    constexpr size_t dataSize = 32768;
    config.SetNumber("tensor_size", dataSize);
    config.SetNumber("shard_size", dataSize);
    config.SetNumber("block_size", dataSize);
    PosixStore store;
    auto s = store.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t nBlocks = 1;
    auto founds = store.Lookup(&block, nBlocks);
    ASSERT_TRUE(founds.HasValue());
    ASSERT_EQ(founds.Value(), std::vector<uint8_t>{false});
    UC::Test::Detail::DataGenerator data1{nBlocks, dataSize};
    data1.GenerateRandom();
    UC::Detail::TaskDesc desc1;
    desc1.brief = "Dump";
    desc1.push_back(UC::Detail::Shard{block, 0, {data1.Buffer()}});
    auto handle1 = store.Dump(desc1);
    ASSERT_TRUE(handle1.HasValue());
    s = store.Wait(handle1.Value());
    ASSERT_EQ(s, UC::Status::OK());
    founds = store.Lookup(&block, nBlocks);
    ASSERT_TRUE(founds.HasValue());
    ASSERT_EQ(founds.Value(), std::vector<uint8_t>{true});
    UC::Test::Detail::DataGenerator data2{nBlocks, dataSize};
    data2.Generate();
    UC::Detail::TaskDesc desc2;
    desc2.brief = "Load";
    desc2.push_back(UC::Detail::Shard{block, 0, {data2.Buffer()}});
    auto handle2 = store.Load(desc2);
    ASSERT_TRUE(handle2.HasValue());
    s = store.Wait(handle2.Value());
    ASSERT_EQ(s, UC::Status::OK());
    ASSERT_EQ(data1.Compare(data2), 0);

    ASSERT_EQ(store.CheckHealth(), UC::Status::OK());
    auto missingBlock =
        UC::Test::Detail::TypesHelper::MakeBlockId("ffffffffffffffffffffffffffffffff");
    UC::Detail::TaskDesc missingDesc;
    missingDesc.brief = "LoadMissing";
    missingDesc.push_back(UC::Detail::Shard{missingBlock, 0, {data2.Buffer()}});
    auto missingHandle = store.Load(missingDesc);
    ASSERT_TRUE(missingHandle.HasValue());
    ASSERT_EQ(store.Wait(missingHandle.Value()), UC::Status::NotFound());
}

TEST_F(UCPosixStoreTest, StartupRejectsBackendIoFailuresWithDetails)
{
    using namespace UC::PosixStore;
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    for (const std::string operation : {"open", "pwrite", "pread", "fsync", "verify"}) {
        SCOPED_TRACE(operation);
        const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / operation);
        const auto failing = std::filesystem::path{Path()} / (operation + "_bad");
        const auto later = std::filesystem::path{Path()} / (operation + "_later");
        std::filesystem::create_directory(shared);
        std::filesystem::create_directory_symlink(shared, failing);
        std::filesystem::create_directory_symlink(shared, later);
        auto config = MakePsyncConfig(shared.string());
        config.Set("storage_backends",
                   std::vector<std::string>{shared.string(), failing.string(), later.string()});
        config.Set("io_direct", operation == "verify");
        UC::Common::StoreHealthConfig health;
        health.enabled = false;
        config.Set("store_health", health);
        size_t opens = 0;
        TestHooks::SetOpenHook([&](const std::string& path, int32_t flags, mode_t mode) {
            ++opens;
            if (path.find(failing.string() + "/") == 0) {
                if (operation == "open") {
                    errno = EROFS;
                    return -1;
                }
                if (operation == "pwrite") { flags = (flags & ~O_ACCMODE) | O_RDONLY; }
                if (operation == "pread") { flags = (flags & ~O_ACCMODE) | O_WRONLY; }
                if (operation == "fsync" || operation == "verify") {
                    return ::open("/dev/zero", O_RDWR);
                }
            }
            return ::open(path.c_str(), flags, mode);
        });
        PosixStore store;
        const auto status = store.Setup(config);
        EXPECT_EQ(status, operation == "verify" ? UC::Status::Error() : UC::Status::OsApiError());
        EXPECT_NE(status.ToString().find(failing.string()), std::string::npos);
        EXPECT_NE(status.ToString().find("failed startup I/O check"), std::string::npos);
        EXPECT_NE(status.ToString().find(operation + "('"), std::string::npos);
        if (operation == "verify") {
            EXPECT_NE(status.ToString().find("health data mismatch"), std::string::npos);
        } else {
            const auto error =
                operation == "open" ? EROFS : (operation == "fsync" ? EINVAL : EBADF);
            EXPECT_NE(status.ToString().find("errno=" + std::to_string(error)), std::string::npos);
            EXPECT_NE(status.ToString().find(std::strerror(error)), std::string::npos);
        }
        EXPECT_EQ(opens, 2);
        EXPECT_EQ(store.CheckHealth(), UC::Status::StoreUnhealthy());
        EXPECT_TRUE(std::filesystem::is_empty(shared / "data"));
        TestHooks::ClearOpenHook();
    }
}

TEST_F(UCPosixStoreTest, CheckHealthWithoutDirectIoOnTemporaryFilesystem)
{
    using namespace UC::PosixStore;
    const auto path =
        std::filesystem::temp_directory_path() / ("ucm_posix_health_" + std::to_string(::getpid()));
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{path};
    std::filesystem::create_directories(path);

    UC::Detail::Dictionary config;
    config.SetNumber("device_id", -1);
    config.Set("storage_backends", std::vector<std::string>{path.string()});
    constexpr size_t dataSize = 4096;
    config.SetNumber("tensor_size", dataSize);
    config.SetNumber("shard_size", dataSize);
    config.SetNumber("block_size", dataSize);
    config.Set("io_direct", false);
    PosixStore store;

    ASSERT_EQ(store.Setup(config), UC::Status::OK());
    EXPECT_EQ(store.CheckHealth(), UC::Status::OK());
    EXPECT_EQ(store.CheckHealth(), UC::Status::OK());
}

TEST_F(UCPosixStoreTest, CheckHealthSucceedsWithEitherStorageBackend)
{
    using namespace UC::PosixStore;
    const auto mount0 = std::filesystem::path{Path()} / "mount0";
    const auto mount1 = std::filesystem::path{Path()} / "mount1";
    std::filesystem::create_directories(mount0);
    std::filesystem::create_directories(mount1 / "data");

    auto config = MakePsyncConfig(mount0.string());
    config.Set("storage_backends", std::vector<std::string>{mount0.string(), mount1.string()});
    PosixStore store;
    ASSERT_EQ(store.Setup(config), UC::Status::OK());
    ASSERT_EQ(store.CheckHealth(), UC::Status::OK());

    const auto checkUnavailable = [&store](const std::filesystem::path& mount) {
        auto unavailable = mount;
        unavailable += ".unavailable";
        std::filesystem::rename(mount, unavailable);
        auto status = store.CheckHealth();
        std::filesystem::rename(unavailable, mount);
        return status;
    };
    EXPECT_TRUE(checkUnavailable(mount0).Success());
    EXPECT_TRUE(checkUnavailable(mount1).Success());
}

TEST_F(UCPosixStoreTest, AioAndPsyncKeepBlockLayersOnTheSameBackend)
{
    using namespace UC::PosixStore;
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    for (const auto& engine : {"aio", "psync"}) {
        SCOPED_TRACE(engine);
        const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / engine);
        const auto mount = std::filesystem::path{Path()} / (std::string(engine) + "_alias");
        std::filesystem::create_directory(shared);
        std::filesystem::create_directory_symlink(shared, mount);
        const std::vector<std::string> backends{shared.string(), mount.string()};
        auto config = MakeAioConfig(shared.string(), 1000, 2, 2);
        config.Set("storage_backends", backends);
        config.Set("posix_io_engine", std::string(engine));
        config.SetNumber("posix_data_trans_concurrency", 2);
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());
        const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        const auto primary = backends[UC::Detail::BlockIdHasher{}(block) % backends.size()] + "/";
        auto opens = std::make_shared<std::atomic<size_t>>(0);
        TestHooks::SetOpenHook(
            [primary, opens](const std::string& path, int32_t flags, mode_t mode) {
                EXPECT_EQ(path.find(primary), 0);
                ++*opens;
                return ::open(path.c_str(), flags, mode);
            });
        auto source0 = MakeAlignedBuffer(0x1234);
        auto source1 = MakeAlignedBuffer(0xabcd);
        auto target = MakeAlignedBuffer(0);
        const std::array<void*, 2> sources{source0.get(), source1.get()};
        for (size_t layer = 0; layer < sources.size(); ++layer) {
            UC::Detail::TaskDesc desc;
            desc.push_back(UC::Detail::Shard{block, layer, {sources[layer]}});
            auto dump = store.Dump(std::move(desc));
            ASSERT_TRUE(dump.HasValue());
            ASSERT_EQ(store.Wait(dump.Value()), UC::Status::OK());
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (store.LookupOnPrefix(&block, 1).Value() != 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        ASSERT_EQ(store.LookupOnPrefix(&block, 1).Value(), 0);
        for (size_t i = 0; i < 4; ++i) {
            const auto layer = i % sources.size();
            UC::Detail::TaskDesc desc;
            desc.push_back(UC::Detail::Shard{block, layer, {target.get()}});
            auto load = store.Load(std::move(desc));
            ASSERT_TRUE(load.HasValue());
            ASSERT_EQ(store.Wait(load.Value()), UC::Status::OK());
            EXPECT_EQ(std::memcmp(sources[layer], target.get(), AIO_TEST_DATA_SIZE), 0);
        }
        EXPECT_EQ(opens->load(), 6);
        TestHooks::ClearOpenHook();
    }
}

TEST_F(UCPosixStoreTest, AioAndPsyncKeepDataAcrossBackendFailureAndRecovery)
{
    using namespace UC::PosixStore;
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    for (const auto& engine : {"aio", "psync"}) {
        SCOPED_TRACE(engine);
        const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} /
                                                      (std::string(engine) + "_shared"));
        const auto mount0 = std::filesystem::path{Path()} / (std::string(engine) + "_mount0");
        const auto mount1 = std::filesystem::path{Path()} / (std::string(engine) + "_mount1");
        std::filesystem::create_directory(shared);
        std::filesystem::create_directory_symlink(shared, mount0);
        std::filesystem::create_directory_symlink(shared, mount1);
        auto config = MakeAioConfig(mount0.string(), 3000);
        config.Set("storage_backends", std::vector<std::string>{mount0.string(), mount1.string()});
        config.Set("posix_io_engine", std::string(engine));
        config.SetNumber("posix_data_trans_concurrency", 2);
        UC::Common::StoreHealthConfig health;
        health.enabled = false;
        health.healthCheckInterval = std::chrono::milliseconds(50);
        health.healthCheckTimeout = std::chrono::milliseconds(40);
        health.healthWindowSize = 4;
        health.failureThreshold = 2;
        config.Set("store_health", health);
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());
        auto buffer = MakeAlignedBuffer(0x12345678);
        auto loaded = MakeAlignedBuffer(0);
        ASSERT_NE(buffer.get(), nullptr);
        ASSERT_NE(loaded.get(), nullptr);
        const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        auto dump = store.Dump(MakeDumpDesc("initial dump", block, buffer.get()));
        ASSERT_TRUE(dump.HasValue());
        ASSERT_EQ(store.Wait(dump.Value()), UC::Status::OK());
        auto waitUntil = [](auto condition) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!condition()) {
                if (std::chrono::steady_clock::now() >= deadline) { return false; }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            return true;
        };
        ASSERT_TRUE(waitUntil([&] { return store.LookupOnPrefix(&block, 1).Value() == 0; }));
        std::filesystem::remove(mount0);
        std::filesystem::remove(mount1);
        std::ofstream(mount0) << "unavailable backend";
        std::ofstream(mount1) << "unavailable backend";
        TestHooks::SetOpenHook([](const std::string& path, int32_t flags, mode_t mode) {
            const auto fd = ::open(path.c_str(), flags, mode);
            if (fd < 0 && errno == ENOTDIR) { errno = EIO; }
            return fd;
        });
        for (size_t i = 0; i < 2; ++i) {
            auto failed = store.Load(MakeDumpDesc("passive failures", block, loaded.get()));
            ASSERT_TRUE(failed.HasValue());
            EXPECT_EQ(store.Wait(failed.Value()), UC::Status::OsApiError());
        }
        ASSERT_EQ(store.CheckHealth(), UC::Status::StoreUnhealthy());
        auto unavailable = store.Load(MakeDumpDesc("no backends", block, loaded.get()));
        ASSERT_TRUE(unavailable.HasValue());
        EXPECT_EQ(store.Wait(unavailable.Value()), UC::Status::StoreUnhealthy());

        std::filesystem::remove(mount1);
        std::filesystem::create_directory_symlink(shared, mount1);
        ASSERT_TRUE(waitUntil([&] { return store.CheckHealth().Success(); }));
        auto load = store.Load(MakeDumpDesc("load through recovered mount", block, loaded.get()));
        ASSERT_TRUE(load.HasValue());
        ASSERT_EQ(store.Wait(load.Value()), UC::Status::OK());
        EXPECT_EQ(std::memcmp(buffer.get(), loaded.get(), AIO_TEST_DATA_SIZE), 0);
        const auto newBlock = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        dump = store.Dump(MakeDumpDesc("dump through healthy mount", newBlock, buffer.get()));
        ASSERT_TRUE(dump.HasValue());
        ASSERT_EQ(store.Wait(dump.Value()), UC::Status::OK());
        ASSERT_TRUE(waitUntil([&] { return store.LookupOnPrefix(&newBlock, 1).Value() == 0; }));
        std::memset(loaded.get(), 0, AIO_TEST_DATA_SIZE);
        load = store.Load(MakeDumpDesc("read new block", newBlock, loaded.get()));
        ASSERT_TRUE(load.HasValue());
        ASSERT_EQ(store.Wait(load.Value()), UC::Status::OK());
        EXPECT_EQ(std::memcmp(buffer.get(), loaded.get(), AIO_TEST_DATA_SIZE), 0);
        TestHooks::ClearOpenHook();
    }
}

TEST_F(UCPosixStoreTest, AioAndPsyncRetryBackendErrorsButNotMissingFiles)
{
    using namespace UC::PosixStore;
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    for (const auto& engine : {"aio", "psync"}) {
        SCOPED_TRACE(engine);
        const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / engine);
        const auto mount = std::filesystem::path{Path()} / (std::string(engine) + "_alias");
        std::filesystem::create_directory(shared);
        std::filesystem::create_directory_symlink(shared, mount);
        auto config = MakeAioConfig(shared.string(), 1000, 2);
        config.Set("posix_io_engine", std::string(engine));
        config.SetNumber("posix_data_trans_concurrency", 2);
        const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        const auto primary = UC::Detail::BlockIdHasher{}(block) % 2;
        std::vector<std::string> backends(2);
        backends[primary] = shared.string();
        backends[(primary + 1) % 2] = mount.string();
        config.Set("storage_backends", backends);
        UC::Common::StoreHealthConfig health;
        health.healthCheckInterval = std::chrono::hours(1);
        config.Set("store_health", health);
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());
        auto failedOpens = std::make_shared<std::atomic<size_t>>(0);
        TestHooks::SetOpenHook(
            [shared, failedOpens](const std::string& path, int32_t flags, mode_t mode) {
                if (path.find(shared.string() + "/") == 0) {
                    ++*failedOpens;
                    errno = EIO;
                    return -1;
                }
                return ::open(path.c_str(), flags, mode);
            });
        auto source = MakeAlignedBuffer(0xfeed);
        auto target = MakeAlignedBuffer(0);
        auto dump = store.Dump(MakeDumpDesc("retry dump", block, source.get()));
        ASSERT_TRUE(dump.HasValue());
        ASSERT_EQ(store.Wait(dump.Value()), UC::Status::OK());
        for (size_t i = 0; i < 6; ++i) {
            auto load = store.Load(MakeDumpDesc("retry load", block, target.get()));
            ASSERT_TRUE(load.HasValue());
            ASSERT_EQ(store.Wait(load.Value()), UC::Status::OK());
            EXPECT_EQ(std::memcmp(source.get(), target.get(), AIO_TEST_DATA_SIZE), 0);
        }
        EXPECT_EQ(failedOpens->load(), 2);
        EXPECT_TRUE(store.CheckHealth().Success());
        TestHooks::ClearOpenHook();
    }
    for (const auto& engine : {"aio", "psync"}) {
        const auto alias = std::filesystem::path{Path()} / (std::string("missing_alias_") + engine);
        std::filesystem::create_directory_symlink(std::filesystem::absolute(Path()), alias);
        auto config = MakeAioConfig(Path(), 1000);
        config.Set("posix_io_engine", std::string(engine));
        config.SetNumber("posix_data_trans_concurrency", 1);
        config.Set("storage_backends", std::vector<std::string>{Path(), alias.string()});
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());
        auto count = std::make_shared<std::atomic<size_t>>(0);
        TestHooks::SetOpenHook([count](const std::string&, int32_t, mode_t) {
            ++*count;
            errno = ENOENT;
            return -1;
        });
        auto target = MakeAlignedBuffer(0);
        const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        for (size_t i = 0; i < 4; ++i) {
            auto load = store.Load(MakeDumpDesc("missing file", block, target.get()));
            ASSERT_TRUE(load.HasValue());
            EXPECT_EQ(store.Wait(load.Value()), UC::Status::NotFound());
        }
        EXPECT_EQ(count->load(), 4);
        EXPECT_TRUE(store.CheckHealth().Success());
        TestHooks::ClearOpenHook();
    }
}

TEST_F(UCPosixStoreTest, AioAndPsyncKeepRoutesForNonTransportErrors)
{
    using namespace UC::PosixStore;
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    for (const auto& engine : {"aio", "psync"}) {
        SCOPED_TRACE(engine);
        const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / engine);
        const auto alias = shared.string() + "_alias";
        std::filesystem::create_directory(shared);
        std::filesystem::create_directory_symlink(shared, alias);
        auto config = MakeAioConfig(shared.string(), 3000, 1);
        config.Set("posix_io_engine", std::string(engine));
        config.SetNumber("posix_data_trans_concurrency", 1);
        config.Set("storage_backends", std::vector<std::string>{shared.string(), alias});
        UC::Common::StoreHealthConfig health;
        health.healthCheckInterval = std::chrono::hours(1);
        health.failureThreshold = 1;
        config.Set("store_health", health);
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());
        auto buffer = MakeAlignedBuffer(0);
        const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        for (const int error : {EACCES, EPERM, ENOSPC, EDQUOT, EINVAL, EMFILE, ESTALE, EROFS, EINTR,
                                ECANCELED, EIO}) {
            SCOPED_TRACE(error);
            std::atomic<size_t> calls{0};
            TestHooks::SetOpenHook([&](const std::string&, int32_t, mode_t) {
                ++calls;
                errno = error;
                return -1;
            });
            auto load = store.Load(MakeDumpDesc("errno propagation", block, buffer.get()));
            ASSERT_TRUE(load.HasValue());
            const auto status = store.Wait(load.Value());
            EXPECT_EQ(status, UC::Status::OsApiError());
            EXPECT_EQ(status.SystemError(), error);
            EXPECT_EQ(calls.load(), 2);
            EXPECT_EQ(store.CheckHealth(),
                      error == EIO ? UC::Status::StoreUnhealthy() : UC::Status::OK());
            TestHooks::ClearOpenHook();
        }
    }
}

TEST_F(UCPosixStoreTest, AioSubmissionErrorsPreserveErrnoWithoutPenalizingRoutes)
{
    using namespace UC::PosixStore;
    auto config = MakeAioConfig(Path(), 3000, 1);
    UC::Common::StoreHealthConfig health;
    health.healthCheckInterval = std::chrono::hours(1);
    health.failureThreshold = 1;
    config.Set("store_health", health);
    PosixStore store;
    ASSERT_EQ(store.Setup(config), UC::Status::OK());
    auto buffer = MakeAlignedBuffer(0);
    ScopedAioHooks hooks;
    for (const int error : {EINVAL, EBADF, EFAULT, ENOSYS, EPERM}) {
        SCOPED_TRACE(error);
        TestHooks::SetAioSubmitHook([error](aio_context_t, int64_t, iocb**) {
            errno = error;
            return -1;
        });
        const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        auto dump = store.Dump(MakeDumpDesc("invalid submission", block, buffer.get()));
        ASSERT_TRUE(dump.HasValue());
        const auto status = store.Wait(dump.Value());
        EXPECT_EQ(status, UC::Status::OsApiError());
        EXPECT_EQ(status.SystemError(), error);
        EXPECT_EQ(store.CheckHealth(), UC::Status::OK());
    }
}

TEST_F(UCPosixStoreTest, AioAndPsyncStopAfterTryingEveryBackend)
{
    using namespace UC::PosixStore;
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    for (const auto& engine : {"aio", "psync"}) {
        SCOPED_TRACE(engine);
        const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / engine);
        std::filesystem::create_directory(shared);
        std::vector<std::string> backends{shared.string()};
        for (size_t i = 1; i < 3; ++i) {
            const auto alias = shared.string() + "_alias" + std::to_string(i);
            std::filesystem::create_directory_symlink(shared, alias);
            backends.push_back(alias);
        }
        auto config = MakeAioConfig(shared.string(), 1000, 1);
        config.Set("posix_io_engine", std::string(engine));
        config.SetNumber("posix_data_trans_concurrency", 1);
        config.Set("storage_backends", backends);
        UC::Common::StoreHealthConfig health;
        health.healthCheckInterval = std::chrono::hours(1);
        health.failureThreshold = 2;
        config.Set("store_health", health);
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());
        const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        const auto primary = UC::Detail::BlockIdHasher{}(block) % backends.size();
        std::mutex mutex;
        std::vector<std::string> paths;
        TestHooks::SetOpenHook([&](const std::string& path, int32_t, mode_t) {
            std::lock_guard<std::mutex> lock(mutex);
            paths.push_back(path);
            errno = EIO;
            return -1;
        });
        auto target = MakeAlignedBuffer(0);
        for (size_t round = 0; round < 2; ++round) {
            auto load = store.Load(MakeDumpDesc("all paths fail", block, target.get()));
            ASSERT_TRUE(load.HasValue());
            EXPECT_EQ(store.Wait(load.Value()), UC::Status::OsApiError());
            EXPECT_EQ(store.CheckHealth().Success(), round == 0);
            std::lock_guard<std::mutex> lock(mutex);
            ASSERT_EQ(paths.size(), (round + 1) * backends.size());
            for (size_t i = 0; i < backends.size(); ++i) {
                const auto& backend = backends[(primary + i) % backends.size()];
                EXPECT_EQ(paths[round * backends.size() + i].find(backend + "/"), 0);
            }
        }
        auto load = store.Load(MakeDumpDesc("no available paths", block, target.get()));
        ASSERT_TRUE(load.HasValue());
        EXPECT_EQ(store.Wait(load.Value()), UC::Status::StoreUnhealthy());
        {
            std::lock_guard<std::mutex> lock(mutex);
            EXPECT_EQ(paths.size(), 2 * backends.size());
        }
        TestHooks::ClearOpenHook();
    }
}

TEST_F(UCPosixStoreTest, AioAndPsyncRecordFailuresWithoutPolling)
{
    using namespace UC::PosixStore;
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    for (const auto& engine : {"aio", "psync"}) {
        SCOPED_TRACE(engine);
        const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / engine);
        const auto alias = shared.string() + "_alias";
        std::filesystem::create_directory(shared);
        std::filesystem::create_directory_symlink(shared, alias);
        auto config = MakeAioConfig(shared.string(), 3000, 1);
        config.Set("posix_io_engine", std::string(engine));
        config.SetNumber("posix_data_trans_concurrency", 1);
        config.Set("storage_backends", std::vector<std::string>{shared.string(), alias});
        UC::Common::StoreHealthConfig health;
        health.healthCheckInterval = std::chrono::hours(1);
        health.failureThreshold = 1;
        config.Set("store_health", health);
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());
        auto attempts = std::make_shared<std::atomic<size_t>>(0);
        TestHooks::SetOpenHook([attempts](const std::string&, int32_t, mode_t) {
            ++*attempts;
            errno = EIO;
            return -1;
        });
        auto buffer = MakeAlignedBuffer(0);
        const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        auto load = store.Load(MakeDumpDesc("unobserved failures", block, buffer.get()));
        ASSERT_TRUE(load.HasValue());
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (store.CheckHealth().Success() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        EXPECT_EQ(attempts->load(), 2);
        EXPECT_EQ(store.CheckHealth(), UC::Status::StoreUnhealthy());
        const auto status = store.Wait(load.Value());
        EXPECT_EQ(status, UC::Status::OsApiError());
        EXPECT_EQ(status.SystemError(), EIO);
        TestHooks::ClearOpenHook();
    }
}

TEST_F(UCPosixStoreTest, AioAndPsyncRetryTimedOutBackendWithoutPolling)
{
    using namespace UC::PosixStore;
    for (const auto& engine : {"aio", "psync"}) {
        SCOPED_TRACE(engine);
        const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / engine);
        const auto mount = std::filesystem::path{Path()} / (std::string(engine) + "_alias");
        std::filesystem::create_directory(shared);
        std::filesystem::create_directory_symlink(shared, mount);
        const auto mount2 = std::filesystem::path{Path()} / (std::string(engine) + "_alias2");
        std::filesystem::create_directory_symlink(shared, mount2);
        auto config = MakeAioConfig(shared.string(), 900, 1);
        config.Set("posix_io_engine", std::string(engine));
        config.SetNumber("posix_data_trans_concurrency", 1);
        const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        const auto primary = UC::Detail::BlockIdHasher{}(block) % 3;
        std::vector<std::string> backends(3);
        backends[primary] = shared.string();
        backends[(primary + 1) % 3] = mount.string();
        backends[(primary + 2) % 3] = mount2.string();
        config.Set("storage_backends", backends);
        UC::Common::StoreHealthConfig health;
        health.healthCheckInterval = std::chrono::hours(1);
        config.Set("store_health", health);
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());
        auto source = MakeAlignedBuffer(0xbeef);
        std::mutex mutex;
        std::condition_variable cv;
        bool release = false;
        bool returned = false;
        bool retried = false;
        TestHooks::SetOpenHook([&](const std::string& path, int32_t flags, mode_t mode) {
            if (path.find(shared.string() + "/") == 0) {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return release; });
                returned = true;
                cv.notify_all();
            } else {
                std::lock_guard<std::mutex> lock(mutex);
                retried = true;
                cv.notify_all();
            }
            return ::open(path.c_str(), flags, mode);
        });
        const auto start = std::chrono::steady_clock::now();
        auto dump = store.Dump(MakeDumpDesc("timed out path", block, source.get()));
        bool completed = false;
        if (dump) {
            std::unique_lock<std::mutex> lock(mutex);
            completed = cv.wait_for(lock, std::chrono::seconds(2), [&] { return retried; });
        }
        const auto status = completed ? store.Wait(dump.Value()) : UC::Status::Timeout();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        {
            std::lock_guard<std::mutex> lock(mutex);
            release = true;
        }
        cv.notify_all();
        {
            std::unique_lock<std::mutex> lock(mutex);
            EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(2), [&] { return returned; }));
        }
        TestHooks::ClearOpenHook();
        EXPECT_TRUE(completed);
        EXPECT_EQ(status, UC::Status::OK());
        EXPECT_GE(elapsed, std::chrono::milliseconds(250));
        EXPECT_LT(elapsed, std::chrono::milliseconds(800));
        auto target = MakeAlignedBuffer(0);
        auto load = store.Load(MakeDumpDesc("read retry result", block, target.get()));
        ASSERT_TRUE(load.HasValue());
        ASSERT_EQ(store.Wait(load.Value()), UC::Status::OK());
        EXPECT_EQ(std::memcmp(source.get(), target.get(), AIO_TEST_DATA_SIZE), 0);
    }
}

TEST_F(UCPosixStoreTest, AioAndPsyncWaitingRequestDoesNotBlockOtherRequests)
{
    using namespace UC::PosixStore;
    for (const auto& engine : {"aio", "psync"}) {
        SCOPED_TRACE(engine);
        const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / engine);
        const auto alias = shared.string() + "_alias";
        std::filesystem::create_directory(shared);
        std::filesystem::create_directory_symlink(shared, alias);
        auto config = MakeAioConfig(shared.string(), 10000, 2);
        config.Set("posix_io_engine", std::string(engine));
        config.SetNumber("posix_data_trans_concurrency", 2);
        config.Set("storage_backends", std::vector<std::string>{shared.string(), alias});
        UC::Common::StoreHealthConfig health;
        health.healthCheckInterval = std::chrono::hours(1);
        config.Set("store_health", health);
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());
        auto source = MakeAlignedBuffer(0xbeef);
        auto target = MakeAlignedBuffer(0);
        std::atomic<bool> first{true};
        std::promise<void> entered;
        std::promise<void> release;
        auto released = release.get_future().share();
        TestHooks::SetOpenHook([&](const std::string& path, int32_t flags, mode_t mode) {
            if (first.exchange(false)) {
                entered.set_value();
                released.wait();
            }
            return ::open(path.c_str(), flags, mode);
        });
        const auto slowBlock = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        auto slow = store.Dump(MakeDumpDesc("stalled request", slowBlock, source.get()));
        const auto opened = entered.get_future().wait_for(std::chrono::seconds(2));
        auto slowWait = std::async(std::launch::async,
                                   [&] { return slow ? store.Wait(slow.Value()) : slow.Error(); });
        auto fast = std::async(std::launch::async, [&] {
            if (!slow) { return slow.Error(); }
            auto pending = store.Check(slow.Value());
            if (!pending) { return pending.Error(); }
            EXPECT_FALSE(pending.Value());
            const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
            auto dump = store.Dump(MakeDumpDesc("concurrent dump", block, source.get()));
            if (!dump) { return dump.Error(); }
            auto status = store.Wait(dump.Value());
            if (status.Failure()) { return status; }
            auto load = store.Load(MakeDumpDesc("concurrent load", block, target.get()));
            return load ? store.Wait(load.Value()) : load.Error();
        });
        const auto ready = fast.wait_for(std::chrono::seconds(2));
        release.set_value();
        const auto fastStatus = fast.get();
        const auto slowStatus = slowWait.get();
        TestHooks::ClearOpenHook();
        EXPECT_EQ(opened, std::future_status::ready);
        EXPECT_EQ(ready, std::future_status::ready);
        EXPECT_EQ(fastStatus, UC::Status::OK());
        EXPECT_EQ(slowStatus, UC::Status::OK());
        EXPECT_EQ(std::memcmp(source.get(), target.get(), AIO_TEST_DATA_SIZE), 0);
    }
}

TEST_F(UCPosixStoreTest, AioAndPsyncDoNotRetryDuringDestruction)
{
    using namespace UC::PosixStore;
    for (const auto& engine : {"aio", "psync"}) {
        SCOPED_TRACE(engine);
        const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / engine);
        const auto alias = shared.string() + "_alias";
        std::filesystem::create_directory(shared);
        std::filesystem::create_directory_symlink(shared, alias);
        auto config = MakeAioConfig(shared.string(), 10000, 1);
        config.Set("posix_io_engine", std::string(engine));
        config.SetNumber("posix_data_trans_concurrency", 1);
        config.Set("storage_backends", std::vector<std::string>{shared.string(), alias});
        UC::Common::StoreHealthConfig health;
        health.healthCheckInterval = std::chrono::hours(1);
        config.Set("store_health", health);
        auto buffer = MakeAlignedBuffer(0);
        auto store = std::make_unique<PosixStore>();
        ASSERT_EQ(store->Setup(config), UC::Status::OK());
        std::atomic<size_t> attempts{0};
        std::promise<void> entered;
        std::promise<void> release;
        auto released = release.get_future().share();
        TestHooks::SetOpenHook([&](const std::string&, int32_t, mode_t) {
            if (attempts.fetch_add(1) == 0) { entered.set_value(); }
            released.wait();
            errno = EIO;
            return -1;
        });
        const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        auto load = store->Load(MakeDumpDesc("destroy pending request", block, buffer.get()));
        const auto opened = entered.get_future().wait_for(std::chrono::seconds(2));
        auto destroyed = std::async(std::launch::async, [&] { store.reset(); });
        const auto pending = destroyed.wait_for(std::chrono::milliseconds(200));
        release.set_value();
        destroyed.get();
        TestHooks::ClearOpenHook();
        EXPECT_TRUE(load.HasValue());
        EXPECT_EQ(opened, std::future_status::ready);
        EXPECT_EQ(pending, std::future_status::timeout);
        EXPECT_EQ(attempts.load(), 1);
    }
}

TEST_F(UCPosixStoreTest, AioAndPsyncRetryCommitFailureOnTheNextBackend)
{
    using namespace UC::PosixStore;
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    for (const auto& engine : {"aio", "psync"}) {
        SCOPED_TRACE(engine);
        const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / engine);
        const auto first = shared.string() + "_first";
        const auto second = shared.string() + "_second";
        std::filesystem::create_directory(shared);
        std::filesystem::create_directory_symlink(shared, first);
        std::filesystem::create_directory_symlink(shared, second);
        const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        const auto primary = UC::Detail::BlockIdHasher{}(block) % 2;
        std::vector<std::string> backends(2);
        backends[primary] = first;
        backends[(primary + 1) % 2] = second;
        auto config = MakeAioConfig(first, 3000, 1);
        config.Set("posix_io_engine", std::string(engine));
        config.SetNumber("posix_data_trans_concurrency", 1);
        config.Set("storage_backends", backends);
        UC::Common::StoreHealthConfig health;
        health.healthCheckInterval = std::chrono::hours(1);
        config.Set("store_health", health);
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());
        std::atomic<size_t> opens{0};
        TestHooks::SetOpenHook([&](const std::string& path, int32_t flags, mode_t mode) {
            ++opens;
            const auto fd = ::open(path.c_str(), flags, mode);
            if (fd >= 0 && path.find(first + "/") == 0) {
                // The open fd remains writable, but commit through this path must fail.
                std::filesystem::remove(first);
                std::ofstream(first) << "unavailable backend";
            }
            return fd;
        });
        auto source = MakeAlignedBuffer(0xcafe);
        auto dump = store.Dump(MakeDumpDesc("commit failover", block, source.get()));
        ASSERT_TRUE(dump.HasValue());
        EXPECT_EQ(store.Wait(dump.Value()), UC::Status::OK());
        EXPECT_EQ(opens.load(), 2);
        TestHooks::ClearOpenHook();
        auto target = MakeAlignedBuffer(0);
        auto load = store.Load(MakeDumpDesc("read committed retry", block, target.get()));
        ASSERT_TRUE(load.HasValue());
        ASSERT_EQ(store.Wait(load.Value()), UC::Status::OK());
        EXPECT_EQ(std::memcmp(source.get(), target.get(), AIO_TEST_DATA_SIZE), 0);
    }
}

TEST_F(UCPosixStoreTest, AioAndPsyncRetryOnlyTheFailedShardWithTimeoutDisabled)
{
    using namespace UC::PosixStore;
    struct ResetHook {
        ~ResetHook() { TestHooks::ClearOpenHook(); }
    } reset;
    for (const auto& engine : {"aio", "psync"}) {
        SCOPED_TRACE(engine);
        const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / engine);
        const auto alias = shared.string() + "_alias";
        std::filesystem::create_directory(shared);
        std::filesystem::create_directory_symlink(shared, alias);
        constexpr size_t blockCount = 32;
        auto config = MakeAioConfig(shared.string(), 0, 4);
        config.Set("posix_io_engine", std::string(engine));
        config.SetNumber("posix_data_trans_concurrency", 4);
        config.Set("storage_backends", std::vector<std::string>{shared.string(), alias});
        UC::Common::StoreHealthConfig health;
        health.healthCheckInterval = std::chrono::hours(1);
        health.failureThreshold = blockCount;
        health.healthWindowSize = blockCount;
        config.Set("store_health", health);
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());
        UC::Detail::BlockId blocks[blockCount];
        for (size_t i = 0; i < blockCount; ++i) {
            do {
                blocks[i] = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
            } while (UC::Detail::BlockIdHasher{}(blocks[i]) % 2 != i % 2);
        }
        std::atomic<size_t> failed{0};
        std::atomic<size_t> succeeded{0};
        TestHooks::SetOpenHook([&](const std::string& path, int32_t flags, mode_t mode) {
            if (path.find(shared.string() + "/") == 0) {
                ++failed;
                errno = EIO;
                return -1;
            }
            ++succeeded;
            return ::open(path.c_str(), flags, mode);
        });
        auto source = MakeAlignedBuffer(0xabcd);
        UC::Detail::TaskDesc desc;
        for (const auto& block : blocks) { desc.push_back({block, 0, {source.get()}}); }
        auto dump = store.Dump(std::move(desc));
        ASSERT_TRUE(dump.HasValue());
        ASSERT_EQ(store.Wait(dump.Value()), UC::Status::OK());
        EXPECT_EQ(failed.load(), blockCount / 2);
        EXPECT_EQ(succeeded.load(), blockCount);
        TestHooks::ClearOpenHook();
    }
}

TEST_F(UCPosixStoreTest, AioRetriesSubmissionFailureAndTimeout)
{
    using namespace UC::PosixStore;
    const auto shared = std::filesystem::absolute(std::filesystem::path{Path()} / "shared");
    const auto mount = std::filesystem::path{Path()} / "alias";
    std::filesystem::create_directory(shared);
    std::filesystem::create_directory_symlink(shared, mount);
    auto config = MakeAioConfig(shared.string(), 1000, 2);
    config.Set("storage_backends", std::vector<std::string>{shared.string(), mount.string()});
    PosixStore store;
    ASSERT_EQ(store.Setup(config), UC::Status::OK());
    auto source = MakeAlignedBuffer(0xbeef);
    auto target = MakeAlignedBuffer(0);
    const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto dump = store.Dump(MakeDumpDesc("initial data", block, source.get()));
    ASSERT_TRUE(dump.HasValue());
    ASSERT_EQ(store.Wait(dump.Value()), UC::Status::OK());
    ScopedAioHooks hooks;
    std::atomic<size_t> calls{0};
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cbs) {
        if (++calls == 1) {
            errno = EIO;
            return -1;
        }
        return static_cast<int32_t>(syscall(SYS_io_submit, ctx, nr, cbs));
    });
    auto load = store.Load(MakeDumpDesc("submit retry", block, target.get()));
    ASSERT_TRUE(load.HasValue());
    ASSERT_EQ(store.Wait(load.Value()), UC::Status::OK());
    EXPECT_EQ(calls.load(), 2);
    EXPECT_EQ(std::memcmp(source.get(), target.get(), AIO_TEST_DATA_SIZE), 0);
    calls = 0;
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cbs) {
        if (++calls == 1) { return static_cast<int32_t>(nr); }
        return static_cast<int32_t>(syscall(SYS_io_submit, ctx, nr, cbs));
    });
    std::memset(target.get(), 0, AIO_TEST_DATA_SIZE);
    load = store.Load(MakeDumpDesc("read timeout retry", block, target.get()));
    ASSERT_TRUE(load.HasValue());
    ASSERT_EQ(store.Wait(load.Value()), UC::Status::OK());
    EXPECT_EQ(calls.load(), 2);
    EXPECT_EQ(std::memcmp(source.get(), target.get(), AIO_TEST_DATA_SIZE), 0);
}

TEST_F(UCPosixStoreTest, DumpThenLoadWithIoDirect)
{
    using namespace UC::PosixStore;
    UC::Detail::Dictionary config;
    config.SetNumber("device_id", 0);
    config.Set("storage_backends", std::vector<std::string>{Path()});
    constexpr size_t dataSize = 32768;
    config.SetNumber("tensor_size", dataSize);
    config.SetNumber("shard_size", dataSize);
    config.SetNumber("block_size", dataSize);
    config.Set("io_direct", true);
    PosixStore store;
    auto s = store.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t nBlocks = 1;
    auto founds = store.Lookup(&block, nBlocks);
    ASSERT_TRUE(founds.HasValue());
    ASSERT_EQ(founds.Value(), std::vector<uint8_t>{false});
    void* buffer1 = nullptr;
    auto ret = posix_memalign(&buffer1, 4096, dataSize);
    ASSERT_EQ(ret, 0);
    *(size_t*)buffer1 = 0xfffffffe;
    UC::Detail::TaskDesc desc1;
    desc1.brief = "Dump";
    desc1.push_back(UC::Detail::Shard{block, 0, {buffer1}});
    auto handle1 = store.Dump(desc1);
    ASSERT_TRUE(handle1.HasValue());
    s = store.Wait(handle1.Value());
    ASSERT_EQ(s, UC::Status::OK());
    founds = store.Lookup(&block, nBlocks);
    ASSERT_TRUE(founds.HasValue());
    ASSERT_EQ(founds.Value(), std::vector<uint8_t>{true});
    void* buffer2 = nullptr;
    ret = posix_memalign(&buffer2, 4096, dataSize);
    ASSERT_EQ(ret, 0);
    *(size_t*)buffer2 = 0x00000001;
    UC::Detail::TaskDesc desc2;
    desc2.brief = "Load";
    desc2.push_back(UC::Detail::Shard{block, 0, {buffer2}});
    auto handle2 = store.Load(desc2);
    ASSERT_TRUE(handle2.HasValue());
    s = store.Wait(handle2.Value());
    ASSERT_EQ(s, UC::Status::OK());
    ASSERT_EQ(*(size_t*)buffer1, *(size_t*)buffer2);
    free(buffer1);
    free(buffer2);
}

TEST_F(UCPosixStoreTest, AioMissingLoadReturnsNotFound)
{
    using namespace UC::PosixStore;
    PosixStore store;
    ASSERT_EQ(store.Setup(MakeAioConfig(Path(), 1000)), UC::Status::OK());

    auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto buffer = MakeAlignedBuffer(0);
    ASSERT_NE(buffer, nullptr);
    auto handle = store.Load(MakeDumpDesc("AioMissingLoad", block, buffer.get()));
    ASSERT_TRUE(handle.HasValue());
    EXPECT_EQ(store.Wait(handle.Value()), UC::Status::NotFound());
}

TEST_F(UCPosixStoreTest, PsyncTruncatedLoadReturnsNotFound)
{
    using namespace UC::PosixStore;
    PosixStore store;
    ASSERT_EQ(store.Setup(MakePsyncConfig(Path())), UC::Status::OK());

    auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto source = MakeAlignedBuffer(1);
    auto target = MakeAlignedBuffer(0);
    ASSERT_NE(source, nullptr);
    ASSERT_NE(target, nullptr);
    auto dump = store.Dump(MakeDumpDesc("PsyncTruncatedDump", block, source.get()));
    ASSERT_TRUE(dump.HasValue());
    ASSERT_EQ(store.Wait(dump.Value()), UC::Status::OK());

    Config layoutConfig;
    layoutConfig.storageBackends = {Path()};
    layoutConfig.dataDirShardBytes = 0;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(layoutConfig), UC::Status::OK());
    std::filesystem::resize_file(
        layout.DataFilePath(layoutConfig.storageBackends.front() + "/", block, false),
        AIO_TEST_DATA_SIZE / 2);

    auto load = store.Load(MakeDumpDesc("PsyncTruncatedLoad", block, target.get()));
    ASSERT_TRUE(load.HasValue());
    EXPECT_EQ(store.Wait(load.Value()), UC::Status::NotFound());
}

TEST_F(UCPosixStoreTest, PsyncDispatchQueueServesManySingleShardLoads)
{
    using namespace UC::PosixStore;
    PosixStore store;
    ASSERT_EQ(store.Setup(MakePsyncConfig(Path())), UC::Status::OK());

    auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    UC::Test::Detail::DataGenerator source{1, AIO_TEST_DATA_SIZE};
    source.GenerateRandom();
    UC::Detail::TaskDesc dump;
    dump.brief = "Dump";
    dump.push_back(UC::Detail::Shard{block, 0, {source.Buffer()}});
    auto dumpHandle = store.Dump(std::move(dump));
    ASSERT_TRUE(dumpHandle.HasValue());
    ASSERT_EQ(store.Wait(dumpHandle.Value()), UC::Status::OK());

    constexpr size_t kTasks = 256;
    std::vector<UC::Test::Detail::DataGenerator> targets;
    std::vector<UC::Detail::TaskHandle> handles;
    targets.reserve(kTasks);
    handles.reserve(kTasks);
    for (size_t i = 0; i < kTasks; ++i) {
        targets.emplace_back(1, AIO_TEST_DATA_SIZE);
        targets.back().Generate();
        UC::Detail::TaskDesc load;
        load.brief = "Load";
        load.push_back(UC::Detail::Shard{block, 0, {targets.back().Buffer()}});
        auto handle = store.Load(std::move(load));
        ASSERT_TRUE(handle.HasValue());
        handles.push_back(handle.Value());
    }
    for (size_t i = 0; i < kTasks; ++i) {
        ASSERT_EQ(store.Wait(handles[i]), UC::Status::OK());
        ASSERT_EQ(source.Compare(targets[i]), 0);
    }
}

TEST_F(UCPosixStoreTest, PsyncDispatchQueueServesConcurrentMultiProducerLoads)
{
    using namespace UC::PosixStore;
    PosixStore store;
    ASSERT_EQ(store.Setup(MakePsyncConfig(Path())), UC::Status::OK());

    auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    UC::Test::Detail::DataGenerator source{1, AIO_TEST_DATA_SIZE};
    source.GenerateRandom();
    UC::Detail::TaskDesc dump;
    dump.brief = "Dump";
    dump.push_back(UC::Detail::Shard{block, 0, {source.Buffer()}});
    auto dumpHandle = store.Dump(std::move(dump));
    ASSERT_TRUE(dumpHandle.HasValue());
    ASSERT_EQ(store.Wait(dumpHandle.Value()), UC::Status::OK());

    constexpr size_t kThreads = 8;
    constexpr size_t kTasksPerThread = 32;
    std::vector<std::vector<UC::Test::Detail::DataGenerator>> targets;
    targets.reserve(kThreads);
    for (size_t i = 0; i < kThreads; ++i) {
        targets.emplace_back();
        targets.back().reserve(kTasksPerThread);
        for (size_t j = 0; j < kTasksPerThread; ++j) {
            targets.back().emplace_back(1, AIO_TEST_DATA_SIZE);
            targets.back().back().Generate();
        }
    }

    std::mutex startMtx;
    std::condition_variable startCv;
    bool started = false;
    std::atomic_size_t failures{0};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (size_t i = 0; i < kThreads; ++i) {
        workers.emplace_back([&, i] {
            {
                std::unique_lock<std::mutex> lock(startMtx);
                startCv.wait(lock, [&] { return started; });
            }
            for (size_t j = 0; j < kTasksPerThread; ++j) {
                UC::Detail::TaskDesc load;
                load.brief = "Load";
                load.push_back(UC::Detail::Shard{block, 0, {targets[i][j].Buffer()}});
                auto handle = store.Load(std::move(load));
                if (!handle.HasValue()) {
                    ++failures;
                    continue;
                }
                if (store.Wait(handle.Value()).Failure()) {
                    ++failures;
                    continue;
                }
                if (source.Compare(targets[i][j]) != 0) { ++failures; }
            }
        });
    }
    {
        std::lock_guard<std::mutex> lock(startMtx);
        started = true;
    }
    startCv.notify_all();
    for (auto& worker : workers) { worker.join(); }
    ASSERT_EQ(failures.load(), 0);
}

TEST_F(UCPosixStoreTest, AioTruncatedLoadReturnsNotFound)
{
    using namespace UC::PosixStore;
    PosixStore store;
    ASSERT_EQ(store.Setup(MakeAioConfig(Path(), 1000)), UC::Status::OK());

    auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto source = MakeAlignedBuffer(1);
    auto target = MakeAlignedBuffer(0);
    ASSERT_NE(source, nullptr);
    ASSERT_NE(target, nullptr);
    auto dump = store.Dump(MakeDumpDesc("AioTruncatedDump", block, source.get()));
    ASSERT_TRUE(dump.HasValue());
    ASSERT_EQ(store.Wait(dump.Value()), UC::Status::OK());

    Config layoutConfig;
    layoutConfig.storageBackends = {Path()};
    layoutConfig.dataDirShardBytes = 0;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(layoutConfig), UC::Status::OK());
    std::filesystem::resize_file(
        layout.DataFilePath(layoutConfig.storageBackends.front() + "/", block, false),
        AIO_TEST_DATA_SIZE / 2);

    auto load = store.Load(MakeDumpDesc("AioTruncatedLoad", block, target.get()));
    ASSERT_TRUE(load.HasValue());
    EXPECT_EQ(store.Wait(load.Value()), UC::Status::NotFound());
}

TEST_F(UCPosixStoreTest, AioWaitTimesOutWhenOpenStalls)
{
    using namespace UC::PosixStore;
    RegisterCounter("posix_aio_timeout_total");
    PosixStore store;
    ASSERT_EQ(store.Setup(MakeAioConfig(Path())), UC::Status::OK());
    StallingOpenHook hook;
    auto buffer = MakeAlignedBuffer(1);
    ASSERT_NE(buffer.get(), nullptr);
    auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto handle = store.Dump(MakeDumpDesc("AioOpenStall", block, buffer.get()));
    ASSERT_TRUE(handle.HasValue());
    ASSERT_TRUE(hook.WaitEntered());

    auto start = std::chrono::steady_clock::now();
    auto status = store.Wait(handle.Value());
    auto elapsed = std::chrono::steady_clock::now() - start;

    ASSERT_EQ(status, UC::Status::Timeout());
    ASSERT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 1000);
    ASSERT_GE(ReadCounter("posix_aio_timeout_total"), 1.0);
}

TEST_F(UCPosixStoreTest, AioWaitTimesOutWhenCompletionIsLost)
{
    using namespace UC::PosixStore;
    PosixStore store;
    ASSERT_EQ(store.Setup(MakeAioConfig(Path())), UC::Status::OK());
    ScopedAioHooks hooks;
    std::atomic<size_t> submits{0};
    TestHooks::SetAioSubmitHook([&submits](aio_context_t, int64_t nr, iocb**) {
        submits.fetch_add(static_cast<size_t>(nr), std::memory_order_relaxed);
        return static_cast<int32_t>(nr);
    });
    TestHooks::SetAioCancelHook([](aio_context_t, struct iocb*, io_event*) { return 0; });
    auto buffer = MakeAlignedBuffer(2);
    ASSERT_NE(buffer.get(), nullptr);
    auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto handle = store.Dump(MakeDumpDesc("AioLostCompletion", block, buffer.get()));
    ASSERT_TRUE(handle.HasValue());

    auto start = std::chrono::steady_clock::now();
    auto status = store.Wait(handle.Value());
    auto elapsed = std::chrono::steady_clock::now() - start;

    ASSERT_EQ(status, UC::Status::Timeout());
    ASSERT_GT(submits.load(std::memory_order_relaxed), 0);
    ASSERT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 1000);
}

TEST_F(UCPosixStoreTest, AioCheckFinishesLostCompletionAfterDeadline)
{
    using namespace UC::PosixStore;
    PosixStore store;
    ASSERT_EQ(store.Setup(MakeAioConfig(Path(), 30)), UC::Status::OK());
    ScopedAioHooks hooks;
    TestHooks::SetAioSubmitHook(
        [](aio_context_t, int64_t nr, iocb**) { return static_cast<int32_t>(nr); });
    TestHooks::SetAioCancelHook([](aio_context_t, struct iocb*, io_event*) { return 0; });
    auto buffer = MakeAlignedBuffer(3);
    ASSERT_NE(buffer.get(), nullptr);
    auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto handle = store.Dump(MakeDumpDesc("AioCheckLostCompletion", block, buffer.get()));
    ASSERT_TRUE(handle.HasValue());

    bool finished = false;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
    while (std::chrono::steady_clock::now() < deadline) {
        auto check = store.Check(handle.Value());
        ASSERT_TRUE(check.HasValue());
        if (check.Value()) {
            finished = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    ASSERT_TRUE(finished);
    ASSERT_EQ(store.Wait(handle.Value()), UC::Status::Timeout());
}

TEST(UCAioImplTest, SubmitEagainHonorsDeadline)
{
    using namespace UC::PosixStore;
    ScopedAioHooks hooks;
    TestHooks::SetAioSubmitHook([](aio_context_t, int64_t, iocb**) {
        errno = EAGAIN;
        return -1;
    });
    AioImpl aio;
    ASSERT_EQ(aio.Setup(30), UC::Status::OK());
    auto buffer = MakeAlignedBuffer(4);
    ASSERT_NE(buffer.get(), nullptr);
    AioImpl::Io io;
    io.fd = 0;
    io.offset = 0;
    io.length = AIO_TEST_DATA_SIZE;
    io.buffer = buffer.get();
    io.callback = [](AioImpl::Result) {};

    auto start = std::chrono::steady_clock::now();
    auto status = aio.ReadAsync(std::move(io));
    EXPECT_EQ(status.SystemError(), EAGAIN);
    auto elapsed = std::chrono::steady_clock::now() - start;

    ASSERT_EQ(status, UC::Status::Timeout());
    ASSERT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 20);
    ASSERT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 1000);
}

TEST_F(UCPosixStoreTest, AioQueuedTasksTimeOutWhileOpenWorkerIsStuck)
{
    using namespace UC::PosixStore;
    PosixStore store;
    ASSERT_EQ(store.Setup(MakeAioConfig(Path(), 50, 1)), UC::Status::OK());
    StallingOpenHook hook;
    auto buffer1 = MakeAlignedBuffer(5);
    auto buffer2 = MakeAlignedBuffer(6);
    auto buffer3 = MakeAlignedBuffer(7);
    ASSERT_NE(buffer1.get(), nullptr);
    ASSERT_NE(buffer2.get(), nullptr);
    ASSERT_NE(buffer3.get(), nullptr);
    auto block1 = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto block2 = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto block3 = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto handle1 = store.Dump(MakeDumpDesc("AioQueuedTimeout1", block1, buffer1.get()));
    ASSERT_TRUE(handle1.HasValue());
    ASSERT_TRUE(hook.WaitEntered());
    auto handle2 = store.Dump(MakeDumpDesc("AioQueuedTimeout2", block2, buffer2.get()));
    auto handle3 = store.Dump(MakeDumpDesc("AioQueuedTimeout3", block3, buffer3.get()));
    ASSERT_TRUE(handle2.HasValue());
    ASSERT_TRUE(handle3.HasValue());

    auto start = std::chrono::steady_clock::now();
    ASSERT_EQ(store.Wait(handle2.Value()), UC::Status::Timeout());
    ASSERT_EQ(store.Wait(handle3.Value()), UC::Status::Timeout());
    ASSERT_EQ(store.Wait(handle1.Value()), UC::Status::Timeout());
    auto elapsed = std::chrono::steady_clock::now() - start;

    ASSERT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 1500);
}

TEST_F(UCPosixStoreTest, AioTimedOutMultiShardDumpDoesNotCommitAfterLateOpen)
{
    using namespace UC::PosixStore;
    PosixStore store;
    ASSERT_EQ(store.Setup(MakeAioConfig(Path(), 50, 1, 2)), UC::Status::OK());
    auto buffer1 = MakeAlignedBuffer(8);
    auto buffer2 = MakeAlignedBuffer(9);
    ASSERT_NE(buffer1.get(), nullptr);
    ASSERT_NE(buffer2.get(), nullptr);
    auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();

    {
        StallingOpenHook hook{true};
        UC::Detail::TaskDesc desc;
        desc.brief = "AioMultiShardLateOpen";
        desc.push_back(UC::Detail::Shard{block, 0, {buffer1.get()}});
        desc.push_back(UC::Detail::Shard{block, 1, {buffer2.get()}});
        auto handle = store.Dump(std::move(desc));
        ASSERT_TRUE(handle.HasValue());
        ASSERT_TRUE(hook.WaitEntered());

        ASSERT_EQ(store.Wait(handle.Value()), UC::Status::Timeout());
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < deadline) {
        auto founds = store.Lookup(&block, 1);
        ASSERT_TRUE(founds.HasValue());
        ASSERT_EQ(founds.Value(), std::vector<uint8_t>{false});
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
