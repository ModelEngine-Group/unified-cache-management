// Regression: real shared memory and native queues, simulated SSD and H2D.
// Real native shared buffer and load queues, two processes; synchronous mock SSD
// and simulator H2D. The split acquisitions model SubmitBounded's legal interleaving.
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <future>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#include <gtest/gtest.h>
#include "cache/cc/early_buffer_pins.h"
#include "cache/cc/early_native_transfer.h"
#include "detail/mock_store.h"
#include "detail/random.h"

namespace {
void Require(bool ok, const char* text) {
    if (!ok) { std::cerr << "REPRO FAILED pid=" << getpid() << " " << text << std::endl; _exit(2); }
}
void Send(int fd) { char byte = 'x'; Require(write(fd, &byte, 1) == 1, "send"); }
void Receive(int fd) {
    pollfd p{fd, POLLIN, 0}; char byte;
    Require(poll(&p, 1, 5000) == 1 && read(fd, &byte, 1) == 1, "receive");
}

void RunScenario(bool crossed) {
    using namespace UC::CacheStore;
    int toChild[2], toParent[2];
    Require(pipe(toChild) == 0 && pipe(toParent) == 0, "pipes");
    Config config;
    config.uniqueId = UC::Test::Detail::Random().RandomString(16);
    config.tensorSizes = {4096}; config.shardSize = 4096; config.blockSize = 4096;
    config.bufferCapacity = 4 << 20; config.loadExclusiveBufferNumber = 16;
    config.shareBufferEnable = true; config.localRankSize = 2;
    config.waitingQueueDepth = 16; config.runningQueueDepth = 16;
    const auto child = fork();
    Require(child >= 0, "fork");
    const int rank = child == 0 ? 1 : 0;
    config.deviceId = rank;
    {
        std::atomic<int> ioCompleted{0}, waits{0};
        class Backend : public testing::NiceMock<UC::Test::Detail::MockStore> {
        public:
            UC::Expected<std::shared_ptr<UC::ReadLease>> AcquireReadLease(
                const UC::Detail::BlockId*, size_t) override
            { return std::make_shared<UC::ReadLease>(); }
        } backend;
        ON_CALL(backend, Check).WillByDefault(testing::Invoke([](auto) { return true; }));
        config.storeBackend = &backend;
        EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc desc) {
            for (auto& shard : desc) { std::memset(shard.addrs[0], 0x6b, 4096); }
            ++ioCompleted;
            return UC::Detail::TaskHandle{1};
        }));
        EXPECT_CALL(backend, Wait).WillOnce(testing::Invoke([&](UC::Detail::TaskHandle) {
            ++waits; return UC::Status::OK();
        }));
        if (rank == 1) { Receive(toChild[0]); }
        TransBuffer buffer;
        Require(buffer.Setup(config).Success(), "shared buffer setup");
        std::cout << "domain rank=" << rank << " backing="
                  << EarlyBufferPins::DescribeDomain(buffer).backing << std::endl;
        if (rank == 0) { Send(toChild[1]); }
        UC::Detail::BlockId a{}, b{}; b[0] = std::byte{1};
        std::array<unsigned char, 8192> destination{};
        UC::Detail::TaskDesc desc{{a, 0, {destination.data()}}, {b, 0, {destination.data() + 4096}}};
        desc.brief = "EarlyConsumer";
        auto task = std::make_shared<TransTask>(TransTask::Type::LOAD, desc);
        task->loadHandles.resize(2);
        auto acquire = [&](size_t index) {
            task->loadHandles[index] = EarlyBufferPins::Acquire(buffer, index == 0 ? a : b, 0, true);
            Require(bool(task->loadHandles[index]), "bounded allocation unexpectedly failed");
        };
        if (rank == 1) {
            acquire(crossed ? 0 : 1); // child owns A in the crossed case, B in control
            Require(task->loadHandles[crossed ? 0 : 1].Owner(), "child ownership");
            Send(toParent[1]); Receive(toChild[0]);
            acquire(crossed ? 1 : 0);
            Send(toParent[1]);
        } else {
            Receive(toParent[0]); acquire(0); acquire(1);
            Require(task->loadHandles[crossed ? 1 : 0].Owner(), "parent ownership");
            Send(toChild[1]); Receive(toParent[0]);
        }
        TransManager native;
        Require(native.Setup(config, &buffer).Success(), "native queues setup");
        EarlyNativeTransfer queue(backend, native, config);
        auto submitted = queue.Submit(EarlyNativeTask{std::move(*task)});
        Require(bool(submitted), "prepared load submit");
        Require(queue.Wait(submitted.Value()).Success(), "prepared load completion");
        Require(waits == 1, "owner read was not waited exactly once");
        for (auto byte : destination) { Require(byte == 0x6b, "simulated H2D data differs"); }
        std::cout << "DONE rank=" << rank << " crossed=" << crossed
                  << " completed_reads=" << ioCompleted << " bytes_verified=" << destination.size()
                  << std::endl;
        if (rank == 1) { Send(toParent[1]); }
        else { Receive(toParent[0]); int status; Require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, "child failed"); }
    }
    if (rank == 1) { _exit(testing::Test::HasFailure() ? 1 : 0); }
    close(toChild[0]); close(toChild[1]); close(toParent[0]); close(toParent[1]);
}
}
TEST(UCEarlyNativeTransferProcessTest, OrderConsistentOwnershipCompletes) { RunScenario(false); }
TEST(UCEarlyNativeTransferProcessTest, CrossedOwnershipCompletesWithoutFifoPublication) { RunScenario(true); }

TEST(UCEarlyNativeTransferTest, FailedHeadDrainsTailWhileAnotherRequestAndDumpAdvance)
{
    using namespace UC::CacheStore;
    class Backend : public testing::NiceMock<UC::Test::Detail::MockStore> {
    public:
        UC::Expected<std::shared_ptr<UC::ReadLease>> AcquireReadLease(
            const UC::Detail::BlockId*, size_t) override
        { return std::make_shared<UC::ReadLease>(); }
    } backend;
    Config c;
    c.uniqueId = UC::Test::Detail::Random().RandomString(16);
    c.deviceId = 0; c.storeBackend = &backend;
    c.tensorSizes = {4096}; c.shardSize = c.blockSize = 4096;
    c.bufferCapacity = 4 << 20; c.shareBufferEnable = false;
    c.loadExclusiveBufferNumber = 0; c.timeoutMs = 100;
    TransBuffer buffer;
    ASSERT_TRUE(buffer.Setup(c).Success());
    TransManager native;
    ASSERT_TRUE(native.Setup(c, &buffer).Success());
    EarlyNativeTransfer queue(backend, native, c);
    std::atomic<size_t> submitted{0}, completed{0};
    std::atomic<bool> releaseTail{false};
    EXPECT_CALL(backend, Load).Times(3).WillRepeatedly(testing::Invoke(
        [&](UC::Detail::TaskDesc desc) {
            for (auto& shard : desc) { std::memset(shard.addrs[0], 0x6b, 4096); }
            return UC::Detail::TaskHandle{++submitted};
        }));
    EXPECT_CALL(backend, Check).WillRepeatedly(testing::Invoke([&](auto id) {
        return id == 1 ? submitted >= 2 : id == 2 ? releaseTail.load() : true;
    }));
    EXPECT_CALL(backend, Wait).WillRepeatedly(testing::Invoke([&](auto id) {
        ++completed;
        return id == 1 ? UC::Status::NotFound() : UC::Status::OK();
    }));
    auto make = [&](size_t begin, size_t count, unsigned char* out, TransTask::Type type) {
        UC::Detail::TaskDesc desc;
        for (size_t i = 0; i < count; ++i) {
            UC::Detail::BlockId block{};
            const auto number = begin + i;
            std::memcpy(block.data(), &number, sizeof(number));
            desc.push_back({block, 0, {out + i * 4096}});
        }
        TransTask t(type, std::move(desc));
        for (const auto& shard : t.desc) {
            t.loadHandles.push_back(EarlyBufferPins::Acquire(buffer, shard.owner, 0, true));
            EXPECT_TRUE(t.loadHandles.back());
        }
        return t;
    };
    std::vector<unsigned char> first(128 * 4096), other(4096, 0), dumped(4096, 0x42);
    auto firstTask = make(0, 128, first.data(), TransTask::Type::LOAD);
    auto tail = firstTask.loadHandles.back();
    auto result = queue.Submit(EarlyNativeTask{std::move(firstTask)});
    ASSERT_TRUE(result);
    auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (submitted < 2 && std::chrono::steady_clock::now() < until) { usleep(1000); }
    ASSERT_EQ(submitted, 2);
    auto waiting = std::async(std::launch::async, [&] { return queue.Wait(result.Value()); });
    EXPECT_EQ(waiting.wait_for(std::chrono::milliseconds(150)), std::future_status::timeout);
    EXPECT_EQ(tail.GetState(), TransBuffer::State::LOADING);
    auto another = queue.Submit(EarlyNativeTask{make(200, 1, other.data(), TransTask::Type::LOAD)});
    ASSERT_TRUE(another);
    EXPECT_TRUE(queue.Wait(another.Value()).Success());
    EXPECT_EQ(other[0], 0x6b);
    EXPECT_CALL(backend, Dump).WillOnce(testing::Invoke([](UC::Detail::TaskDesc desc) {
        EXPECT_EQ(*static_cast<unsigned char*>(desc[0].addrs[0]), 0x42);
        return UC::Detail::TaskHandle{4};
    }));
    auto dump = queue.Submit(EarlyNativeTask{make(201, 1, dumped.data(), TransTask::Type::DUMP)});
    ASSERT_TRUE(dump);
    EXPECT_TRUE(queue.Wait(dump.Value()).Success());
    EXPECT_FALSE(queue.Check(result.Value()).HasValue()); // Wait owns this receipt
    releaseTail = true;
    EXPECT_EQ(waiting.get(), UC::Status::Timeout());
    EXPECT_TRUE(tail.Ready()); // Failed request cannot poison its completed tail
    EXPECT_EQ(first[0], 0);   // No H2D was submitted for the failed complete task
    tail = {};
}
