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
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "detail/data_generator.h"
#include "detail/path_base.h"
#include "detail/types_helper.h"
#include "metrics_api.h"

class UCPosixStoreTest : public UC::Test::Detail::PathBase {};

TEST(UCPosixConfigTest, KeepsPsyncAsDefaultWithOptionalAioSettings)
{
    const UC::PosixStore::Config config;
    EXPECT_EQ(config.ioEngine, "psync");
    EXPECT_EQ(config.aioQueueDepth, 1024);
    EXPECT_EQ(config.loadHandleCacheSize, 8192);
}

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
    // An empty stalledPrefix stalls every open.
    explicit StallingOpenHook(bool succeedAfterRelease = false, std::string stalledPrefix = {})
        : succeedAfterRelease_{succeedAfterRelease}, stalledPrefix_{std::move(stalledPrefix)}
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
        if (stalledPrefix_.empty() || path.rfind(stalledPrefix_, 0) == 0) {
            return Stall(path, flags, mode);
        } else {
            return ::open(path.c_str(), flags, mode);
        }
    }
    int32_t Stall(const std::string& path, int32_t flags, mode_t mode)
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
    std::string stalledPrefix_;
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

TEST_F(UCPosixStoreTest, OutOfOrderShardsCommitTogetherAndSupportCachedLoads)
{
    using namespace UC::PosixStore;
    for (const std::string engine : {"psync", "aio"}) {
        SCOPED_TRACE(engine);
        const auto path = (std::filesystem::path{Path()} / engine).string();
        std::filesystem::create_directories(path);
        auto config = MakeAioConfig(path, 2000, 1, 3);
        config.Set("posix_io_engine", engine);
        config.Set("io_direct", engine == "aio");
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());

        auto source0 = MakeAlignedBuffer(11);
        auto source1 = MakeAlignedBuffer(22);
        auto source2 = MakeAlignedBuffer(33);
        ASSERT_NE(source0.get(), nullptr);
        ASSERT_NE(source1.get(), nullptr);
        ASSERT_NE(source2.get(), nullptr);
        void* sources[] = {source0.get(), source1.get(), source2.get()};
        const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();

        for (size_t index : {size_t{2}, size_t{0}, size_t{1}}) {
            UC::Detail::TaskDesc desc;
            desc.brief = "OutOfOrderDump";
            desc.push_back(UC::Detail::Shard{block, index, {sources[index]}});
            auto task = store.Dump(std::move(desc));
            ASSERT_TRUE(task.HasValue());
            ASSERT_EQ(store.Wait(task.Value()), UC::Status::OK());

            const bool complete = index == 1;
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(complete ? 2000 : 100);
            bool visible = false;
            do {
                auto found = store.Lookup(&block, 1);
                ASSERT_TRUE(found.HasValue());
                visible = found.Value().front();
                if (complete && visible) { break; }
                ASSERT_FALSE(visible);
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            } while (std::chrono::steady_clock::now() < deadline);
            ASSERT_EQ(visible, complete);
        }

        auto target = MakeAlignedBuffer(0);
        ASSERT_NE(target.get(), nullptr);
        for (size_t repeat = 0; repeat < 2; ++repeat) {
            for (size_t index = 0; index < 3; ++index) {
                std::memset(target.get(), 0, AIO_TEST_DATA_SIZE);
                UC::Detail::TaskDesc desc;
                desc.brief = "CachedShardLoad";
                desc.push_back(UC::Detail::Shard{block, index, {target.get()}});
                auto task = store.Load(std::move(desc));
                ASSERT_TRUE(task.HasValue());
                ASSERT_EQ(store.Wait(task.Value()), UC::Status::OK());
                EXPECT_EQ(std::memcmp(target.get(), sources[index], AIO_TEST_DATA_SIZE), 0);
            }
        }
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
    config.Set("posix_io_engine", std::string("psync"));
    PosixStore store;

    ASSERT_EQ(store.Setup(config), UC::Status::OK());
    EXPECT_EQ(store.CheckHealth(), UC::Status::OK());
    EXPECT_EQ(store.CheckHealth(), UC::Status::OK());
}

TEST_F(UCPosixStoreTest, CheckHealthReportsAvailableBackends)
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
        layout.ArchivedFilePath(layoutConfig.storageBackends.front() + "/", block),
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
    // Commit runs after Wait returns, so wait for the archived file to show up.
    const auto file = layout.ArchivedFilePath(layoutConfig.storageBackends.front() + "/", block);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!std::filesystem::exists(file) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(std::filesystem::exists(file));
    std::filesystem::resize_file(file, AIO_TEST_DATA_SIZE / 2);

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
    auto elapsed = std::chrono::steady_clock::now() - start;

    ASSERT_EQ(status, UC::Status::Timeout());
    ASSERT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 20);
    ASSERT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 1000);
}

TEST(UCAioImplTest, TeardownReturnsLoadHandlesToTheirOwner)
{
    using namespace UC::PosixStore;
    ScopedAioHooks hooks;
    TestHooks::SetAioSubmitHook(
        [](aio_context_t, int64_t nr, iocb**) { return static_cast<int32_t>(nr); });
    for (size_t capacity : {size_t{0}, size_t{1}}) {
        SCOPED_TRACE(capacity);
        LoadHandleCache cache;
        ASSERT_EQ(cache.Setup(capacity, false), UC::Status::OK());
        auto borrowed = cache.GetOrOpen({}, "/dev/null");
        ASSERT_TRUE(borrowed.Valid());
        auto fd = borrowed.Fd();
        auto buffer = MakeAlignedBuffer(8);
        ASSERT_NE(buffer.get(), nullptr);
        {
            AioImpl aio;
            ASSERT_EQ(aio.Setup(0), UC::Status::OK());
            AioImpl::Io io;
            io.fd = fd;
            io.offset = 0;
            io.length = AIO_TEST_DATA_SIZE;
            io.buffer = buffer.get();
            io.tag = 1;
            auto pin = std::make_shared<LoadHandleCache::BorrowedFd>(std::move(borrowed));
            io.callback = [pin](AioImpl::Result) { FAIL() << "No completion was submitted"; };
            ASSERT_EQ(aio.ReadAsync(std::move(io)), UC::Status::OK());
        }
        EXPECT_EQ(::fcntl(fd, F_GETFD) >= 0, capacity != 0);
        if (capacity != 0) {
            auto hit = cache.GetOrOpen({}, "/dev/null");
            EXPECT_TRUE(hit.Cached());
            EXPECT_EQ(hit.Fd(), fd);
        }
    }
}

TEST(UCAioImplTest, TeardownClosesRawDumpHandles)
{
    using namespace UC::PosixStore;
    ScopedAioHooks hooks;
    TestHooks::SetAioSubmitHook(
        [](aio_context_t, int64_t nr, iocb**) { return static_cast<int32_t>(nr); });
    auto buffer = MakeAlignedBuffer(9);
    ASSERT_NE(buffer.get(), nullptr);
    auto fd = ::open("/dev/null", O_WRONLY);
    ASSERT_GE(fd, 0);
    {
        AioImpl aio;
        ASSERT_EQ(aio.Setup(0), UC::Status::OK());
        AioImpl::Io io;
        io.fd = fd;
        io.offset = 0;
        io.length = AIO_TEST_DATA_SIZE;
        io.buffer = buffer.get();
        io.tag = 1;
        io.callback = [](AioImpl::Result) { FAIL() << "No completion was submitted"; };
        ASSERT_EQ(aio.WriteAsync(std::move(io)), UC::Status::OK());
    }
    EXPECT_EQ(::fcntl(fd, F_GETFD), -1);
    EXPECT_EQ(errno, EBADF);
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

    // Dump does not retry, so the queued ones report their own queued timeout.
    auto start = std::chrono::steady_clock::now();
    ASSERT_EQ(store.Wait(handle2.Value()), UC::Status::TimeoutInQueue());
    ASSERT_EQ(store.Wait(handle3.Value()), UC::Status::TimeoutInQueue());
    ASSERT_EQ(store.Wait(handle1.Value()), UC::Status::Timeout());
    auto elapsed = std::chrono::steady_clock::now() - start;

    ASSERT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 1500);
    // Only the task stuck in open counts against the backend; the two queued ones do not,
    // so one failure stays below the passive threshold of 2.
    EXPECT_TRUE(store.CheckHealth().Success());
}

TEST_F(UCPosixStoreTest, AioStuckOpenWorkerIsReplaced)
{
    using namespace UC::PosixStore;
    // Stall only the first open. Later opens go to the filesystem.
    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    std::atomic<size_t> calls{0};
    std::atomic<bool> stalledReturned{false};
    TestHooks::SetOpenHook([&](const std::string& path, int32_t flags, mode_t mode) {
        if (calls.fetch_add(1) == 0) {
            std::unique_lock<std::mutex> lock{mutex};
            cv.wait(lock, [&] { return release; });
            lock.unlock();
            auto fd = ::open(path.c_str(), flags, mode);
            stalledReturned.store(true);
            return fd;
        }
        return ::open(path.c_str(), flags, mode);
    });
    {
        PosixStore store;
        ASSERT_EQ(store.Setup(MakeAioConfig(Path(), 200, 1)), UC::Status::OK());
        auto buffer1 = MakeAlignedBuffer(11);
        auto buffer2 = MakeAlignedBuffer(12);
        ASSERT_NE(buffer1.get(), nullptr);
        ASSERT_NE(buffer2.get(), nullptr);
        auto block1 = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        auto block2 = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();

        auto handle1 = store.Dump(MakeDumpDesc("AioStuckOpen1", block1, buffer1.get()));
        ASSERT_TRUE(handle1.HasValue());
        EXPECT_EQ(store.Wait(handle1.Value()), UC::Status::Timeout());

        // The only open worker is still stuck. The pool replaces it once it runs past the
        // 200 ms timeout, which the monitor notices within its 1 s interval.
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        auto handle2 = store.Dump(MakeDumpDesc("AioStuckOpen2", block2, buffer2.get()));
        ASSERT_TRUE(handle2.HasValue());
        EXPECT_EQ(store.Wait(handle2.Value()), UC::Status::OK());

        // The stuck worker was detached by the pool. Let its open return and give its callback
        // time to finish before the store goes away.
        {
            std::lock_guard<std::mutex> lock{mutex};
            release = true;
        }
        cv.notify_all();
        for (size_t i = 0; i < 100 && !stalledReturned.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    TestHooks::ClearOpenHook();
}

TEST_F(UCPosixStoreTest, AioLoadFailsOverWhenOneBackendStallsOpenWorkers)
{
    using namespace UC::PosixStore;
    // Both backends are paths to the same directory; opens through the second one hang.
    const auto stalledBackend = Path() + "./";
    constexpr size_t timeoutMs = 4000;
    auto config = MakeAioConfig(Path(), timeoutMs, 4);
    config.Set("storage_backends", std::vector<std::string>{Path(), stalledBackend});
    constexpr size_t nBlocks = 32;
    std::vector<UC::Detail::BlockId> blocks;
    std::vector<BufferPtr> buffers;
    const auto makeDesc = [&](const char* brief, const std::vector<UC::Detail::BlockId>& ids) {
        UC::Detail::TaskDesc desc;
        desc.brief = brief;
        for (const auto& block : ids) {
            buffers.push_back(MakeAlignedBuffer(buffers.size() + 1));
            desc.push_back(UC::Detail::Shard{block, 0, {buffers.back().get()}});
        }
        return desc;
    };
    for (size_t i = 0; i < nBlocks; ++i) {
        blocks.push_back(UC::Test::Detail::TypesHelper::MakeBlockIdRandomly());
    }
    {
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());
        auto dump = store.Dump(makeDesc("Dump", blocks));
        ASSERT_TRUE(dump.HasValue());
        ASSERT_EQ(store.Wait(dump.Value()), UC::Status::OK());
    }
    // Commit runs after Wait returns, so wait for the archived files to show up.
    Config layoutConfig;
    layoutConfig.storageBackends = {Path()};
    layoutConfig.dataDirShardBytes = 0;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(layoutConfig), UC::Status::OK());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (const auto& block : blocks) {
        const auto file = layout.ArchivedFilePath(Path(), block);
        while (!std::filesystem::exists(file) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ASSERT_TRUE(std::filesystem::exists(file));
    }

    // A fresh store starts with both backends healthy, so the stuck backend fills all open
    // workers before it trips.
    const auto loadedFrom = buffers.size();
    {
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::OK());
        {
            StallingOpenHook hook{false, stalledBackend};
            const auto start = std::chrono::steady_clock::now();
            auto load = store.Load(makeDesc("Load", blocks));
            ASSERT_TRUE(load.HasValue());
            EXPECT_EQ(store.Wait(load.Value()), UC::Status::OK());
            EXPECT_LT(std::chrono::steady_clock::now() - start,
                      std::chrono::milliseconds(timeoutMs));
        }
        // Let the released open workers finish their callbacks before the store goes away.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    for (size_t i = 0; i < nBlocks; ++i) {
        EXPECT_EQ(*reinterpret_cast<size_t*>(buffers[loadedFrom + i].get()), i + 1);
    }
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
