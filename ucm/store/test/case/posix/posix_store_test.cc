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
#include <memory>
#include <mutex>
#include <thread>
#include "detail/data_generator.h"
#include "detail/path_base.h"
#include "detail/types_helper.h"
#include "metrics_api.h"

class UCPosixStoreTest : public UC::Test::Detail::PathBase {};

namespace {
constexpr size_t AIO_TEST_DATA_SIZE = 4096;
using BufferPtr = std::unique_ptr<void, decltype(&std::free)>;

UC::Detail::Dictionary MakeAioConfig(const std::string& path, size_t timeoutMs = 50,
                                     size_t openConcurrency = 1)
{
    UC::Detail::Dictionary config;
    config.SetNumber("device_id", 0);
    config.Set("storage_backends", std::vector<std::string>{path});
    config.SetNumber("tensor_size", AIO_TEST_DATA_SIZE);
    config.SetNumber("shard_size", AIO_TEST_DATA_SIZE);
    config.SetNumber("block_size", AIO_TEST_DATA_SIZE);
    config.Set("posix_io_engine", std::string("aio"));
    config.SetNumber("timeout_ms", timeoutMs);
    config.SetNumber("posix_open_concurrency", openConcurrency);
    config.SetNumber("posix_commit_concurrency", size_t(1));
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

UC::Detail::TaskDesc MakeLoadDesc(const char* brief, const UC::Detail::BlockId& block, void* buffer)
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
    StallingOpenHook()
    {
        UC::PosixStore::TestHooks::SetOpenHook(
            [this](const std::string&, int32_t, mode_t) { return Run(); });
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
    int32_t Run()
    {
        {
            std::lock_guard<std::mutex> lock{mutex_};
            ++active_;
            entered_ = true;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lock{mutex_};
        cv_.wait(lock, [this] { return release_; });
        --active_;
        cv_.notify_all();
        errno = EIO;
        return -1;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_{false};
    bool release_{false};
    size_t active_{0};
};

class ScopedAioHooks {
public:
    ~ScopedAioHooks() { UC::PosixStore::TestHooks::ClearAioHooks(); }
};

class ControlledProbeHook {
public:
    ControlledProbeHook()
    {
        UC::PosixStore::TestHooks::SetBackendProbeHook(
            [this](const std::vector<std::string>&) { return Run(); });
    }
    ~ControlledProbeHook() { UC::PosixStore::TestHooks::ClearBackendProbeHook(); }
    void Recover()
    {
        std::lock_guard<std::mutex> lock{mutex_};
        recovered_ = true;
    }

private:
    UC::Status Run()
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return recovered_ ? UC::Status::OK() : UC::Status::Timeout();
    }

private:
    std::mutex mutex_;
    bool recovered_{false};
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

TEST_F(UCPosixStoreTest, AioBackendPauseShortCircuitsLoadDumpAndLookup)
{
    using namespace UC::PosixStore;
    ControlledProbeHook probe;
    PosixStore store;
    ASSERT_EQ(store.Setup(MakeAioConfig(Path())), UC::Status::OK());

    auto cachedBuffer = MakeAlignedBuffer(11);
    ASSERT_NE(cachedBuffer.get(), nullptr);
    auto cachedBlock = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto cachedDump =
        store.Dump(MakeDumpDesc("AioCachedBeforePause", cachedBlock, cachedBuffer.get()));
    ASSERT_TRUE(cachedDump.HasValue());
    ASSERT_EQ(store.Wait(cachedDump.Value()), UC::Status::OK());
    auto foundBeforePause = store.Lookup(&cachedBlock, 1);
    ASSERT_TRUE(foundBeforePause.HasValue());
    ASSERT_EQ(foundBeforePause.Value(), std::vector<uint8_t>{true});

    StallingOpenHook hook;
    auto stalledBuffer = MakeAlignedBuffer(12);
    ASSERT_NE(stalledBuffer.get(), nullptr);
    auto stalledBlock = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto stalledDump =
        store.Dump(MakeDumpDesc("AioPauseTrigger", stalledBlock, stalledBuffer.get()));
    ASSERT_TRUE(stalledDump.HasValue());
    ASSERT_TRUE(hook.WaitEntered());
    ASSERT_EQ(store.Wait(stalledDump.Value()), UC::Status::Timeout());

    auto lookupWhilePaused = store.Lookup(&cachedBlock, 1);
    ASSERT_TRUE(lookupWhilePaused.HasValue());
    ASSERT_EQ(lookupWhilePaused.Value(), std::vector<uint8_t>{false});

    auto loadBuffer = MakeAlignedBuffer(13);
    ASSERT_NE(loadBuffer.get(), nullptr);
    auto pausedLoad = store.Load(MakeLoadDesc("AioPausedLoad", cachedBlock, loadBuffer.get()));
    ASSERT_FALSE(pausedLoad.HasValue());
    ASSERT_EQ(pausedLoad.Error(), UC::Status::Timeout());

    auto pausedDumpBlock = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto pausedDumpBuffer = MakeAlignedBuffer(14);
    ASSERT_NE(pausedDumpBuffer.get(), nullptr);
    auto pausedDump =
        store.Dump(MakeDumpDesc("AioPausedDump", pausedDumpBlock, pausedDumpBuffer.get()));
    ASSERT_FALSE(pausedDump.HasValue());
    ASSERT_EQ(pausedDump.Error(), UC::Status::Timeout());
}

TEST_F(UCPosixStoreTest, AioBackendPauseIsSharedByStoresOnSameBackend)
{
    using namespace UC::PosixStore;
    ControlledProbeHook probe;
    auto path = Path();
    PosixStore writer;
    PosixStore lookup;
    ASSERT_EQ(writer.Setup(MakeAioConfig(path)), UC::Status::OK());
    ASSERT_EQ(lookup.Setup(MakeAioConfig(path)), UC::Status::OK());

    auto cachedBuffer = MakeAlignedBuffer(31);
    ASSERT_NE(cachedBuffer.get(), nullptr);
    auto cachedBlock = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto cachedDump =
        writer.Dump(MakeDumpDesc("AioSharedHealthCached", cachedBlock, cachedBuffer.get()));
    ASSERT_TRUE(cachedDump.HasValue());
    ASSERT_EQ(writer.Wait(cachedDump.Value()), UC::Status::OK());
    auto foundBeforePause = lookup.Lookup(&cachedBlock, 1);
    ASSERT_TRUE(foundBeforePause.HasValue());
    ASSERT_EQ(foundBeforePause.Value(), std::vector<uint8_t>{true});

    StallingOpenHook hook;
    auto stalledBuffer = MakeAlignedBuffer(32);
    ASSERT_NE(stalledBuffer.get(), nullptr);
    auto stalledBlock = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto stalledDump =
        writer.Dump(MakeDumpDesc("AioSharedHealthTrigger", stalledBlock, stalledBuffer.get()));
    ASSERT_TRUE(stalledDump.HasValue());
    ASSERT_TRUE(hook.WaitEntered());
    ASSERT_EQ(writer.Wait(stalledDump.Value()), UC::Status::Timeout());

    auto foundAfterPause = lookup.Lookup(&cachedBlock, 1);
    ASSERT_TRUE(foundAfterPause.HasValue());
    ASSERT_EQ(foundAfterPause.Value(), std::vector<uint8_t>{false});
}

TEST_F(UCPosixStoreTest, AioBackendPauseRecoversAfterProbeSucceeds)
{
    using namespace UC::PosixStore;
    ControlledProbeHook probe;
    PosixStore store;
    ASSERT_EQ(store.Setup(MakeAioConfig(Path(), 50)), UC::Status::OK());

    auto cachedBuffer = MakeAlignedBuffer(21);
    ASSERT_NE(cachedBuffer.get(), nullptr);
    auto cachedBlock = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto cachedDump =
        store.Dump(MakeDumpDesc("AioRecoveryCached", cachedBlock, cachedBuffer.get()));
    ASSERT_TRUE(cachedDump.HasValue());
    ASSERT_EQ(store.Wait(cachedDump.Value()), UC::Status::OK());

    {
        StallingOpenHook hook;
        auto stalledBuffer = MakeAlignedBuffer(22);
        ASSERT_NE(stalledBuffer.get(), nullptr);
        auto stalledBlock = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        auto stalledDump =
            store.Dump(MakeDumpDesc("AioRecoveryTrigger", stalledBlock, stalledBuffer.get()));
        ASSERT_TRUE(stalledDump.HasValue());
        ASSERT_TRUE(hook.WaitEntered());
        ASSERT_EQ(store.Wait(stalledDump.Value()), UC::Status::Timeout());
        auto pausedLookup = store.Lookup(&cachedBlock, 1);
        ASSERT_TRUE(pausedLookup.HasValue());
        ASSERT_EQ(pausedLookup.Value(), std::vector<uint8_t>{false});
    }

    probe.Recover();
    bool recovered = false;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
    while (std::chrono::steady_clock::now() < deadline) {
        auto lookup = store.Lookup(&cachedBlock, 1);
        ASSERT_TRUE(lookup.HasValue());
        if (lookup.Value() == std::vector<uint8_t>{true}) {
            recovered = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ASSERT_TRUE(recovered);

    auto loadBuffer = MakeAlignedBuffer(23);
    ASSERT_NE(loadBuffer.get(), nullptr);
    auto load = store.Load(MakeLoadDesc("AioRecoveredLoad", cachedBlock, loadBuffer.get()));
    ASSERT_TRUE(load.HasValue());
    ASSERT_EQ(store.Wait(load.Value()), UC::Status::OK());
    ASSERT_EQ(*reinterpret_cast<size_t*>(loadBuffer.get()), size_t{21});
}

TEST_F(UCPosixStoreTest, AioWaitTimesOutWhenCompletionIsLost)
{
    using namespace UC::PosixStore;
    ControlledProbeHook probe;
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

    auto nextBuffer = MakeAlignedBuffer(24);
    ASSERT_NE(nextBuffer.get(), nullptr);
    auto nextBlock = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto nextDump =
        store.Dump(MakeDumpDesc("AioLostCompletionPaused", nextBlock, nextBuffer.get()));
    ASSERT_FALSE(nextDump.HasValue());
    ASSERT_EQ(nextDump.Error(), UC::Status::Timeout());
}

TEST_F(UCPosixStoreTest, AioSubmitTimeoutPausesBackend)
{
    using namespace UC::PosixStore;
    ControlledProbeHook probe;
    PosixStore store;
    ASSERT_EQ(store.Setup(MakeAioConfig(Path(), 30)), UC::Status::OK());
    ScopedAioHooks hooks;
    TestHooks::SetAioSubmitHook([](aio_context_t, int64_t, iocb**) {
        errno = EAGAIN;
        return -1;
    });

    auto buffer = MakeAlignedBuffer(25);
    ASSERT_NE(buffer.get(), nullptr);
    auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto handle = store.Dump(MakeDumpDesc("AioSubmitTimeoutPause", block, buffer.get()));
    ASSERT_TRUE(handle.HasValue());
    ASSERT_EQ(store.Wait(handle.Value()), UC::Status::Timeout());

    auto nextBuffer = MakeAlignedBuffer(26);
    ASSERT_NE(nextBuffer.get(), nullptr);
    auto nextBlock = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto nextDump = store.Dump(MakeDumpDesc("AioSubmitTimeoutPaused", nextBlock, nextBuffer.get()));
    ASSERT_FALSE(nextDump.HasValue());
    ASSERT_EQ(nextDump.Error(), UC::Status::Timeout());
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
