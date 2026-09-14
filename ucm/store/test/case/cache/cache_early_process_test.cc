// Native shared L1 and load queues in two processes; backend/HBM are simulated.
#include <array>
#include <cstring>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#include <gtest/gtest.h>
#include "cache/cc/load_queue.h"
#include "cache/cc/reserved_prefetch.h"
#include "detail/mock_store.h"
#include "detail/random.h"

namespace {
class Backend : public testing::NiceMock<UC::Test::Detail::MockStore> {
public:
    UC::Expected<std::shared_ptr<UC::ReadLease>> AcquireReadLease(
        const UC::Detail::BlockId*, size_t) override
    { return std::make_shared<UC::ReadLease>(); }
};

void Require(bool condition, const char* reason)
{
    if (!condition) { throw std::runtime_error(reason); }
}

void Send(int fd)
{
    char value = 'x';
    Require(::write(fd, &value, 1) == 1, "pipe send");
}

void Receive(int fd)
{
    pollfd item{fd, POLLIN, 0};
    Require(::poll(&item, 1, 5000) == 1, "pipe timeout");
    char value;
    Require(::read(fd, &value, 1) == 1, "pipe receive");
}

void LoadAndRelease(UC::CacheStore::ReservedPrefetch& early,
                    UC::CacheStore::BufferManager& manager,
                    const UC::CacheStore::Config& config, uint64_t reservation,
                    const UC::Detail::BlockId& block)
{
    std::array<unsigned char, 8192> destination{};
    UC::Detail::TaskDesc desc{
        {block, 0, {destination.data()}}, {block, 1, {destination.data() + 4096}}};
    desc.loadReservations = {reservation};
    auto task = std::make_shared<UC::CacheStore::TransTask>(
        UC::CacheStore::TransTask::Type::LOAD, desc);
    task->loadHandles = early.LoadHandles(desc);
    Require(task->loadHandles.size() == 2, "local handoff");
    early.Release(reservation);
    UC::HashSet<UC::Detail::TaskHandle> failures;
    UC::CacheStore::LoadQueue queue;
    Require(queue.Setup(config, &failures, manager.GetTransBuffer()).Success(), "load setup");
    auto waiter = std::make_shared<UC::Latch>();
    queue.Submit(task, waiter);
    Require(waiter->WaitForDuration(5000), "load timeout");
    Require(!failures.Contains(task->id), "load failed");
    for (auto byte : destination) { Require(byte == 0x6b, "shared bytes differ"); }
}
}  // namespace

TEST(UCCacheEarlyProcessTest, SharedShardHasOneReaderAndTwoLocalHandoffs)
{
    int started[2], reserved[2];
    ASSERT_EQ(::pipe(started), 0);
    ASSERT_EQ(::pipe(reserved), 0);
    UC::CacheStore::Config config;
    config.uniqueId = UC::Test::Detail::Random().RandomString(16);
    config.deviceId = 0;
    config.tensorSizes = {4096};
    config.shardSize = 4096;
    config.blockSize = 8192;
    config.bufferCapacity = 4 << 20;
    config.loadExclusiveBufferNumber = 0;
    config.shareBufferEnable = true;
    const auto child = ::fork();  // before either process starts native threads
    ASSERT_GE(child, 0);
    auto run = [&]() {
        Backend backend;
        config.storeBackend = &backend;
        if (child == 0) {
            config.deviceId = 1;
            Receive(started[0]);
        }
        UC::CacheStore::BufferManager manager;
        Require(manager.Setup(config).Success(), "shared buffer setup");
        UC::CacheStore::ReservedPrefetch early(
            &manager, &backend, config.blockSize, config.shardSize, config.timeoutMs);
        EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(ssize_t{0}));
        if (child == 0) {
            EXPECT_CALL(backend, Load).Times(0);
            EXPECT_CALL(backend, Wait).Times(0);
        } else {
            EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc task) {
                EXPECT_EQ(task.size(), 2);
                for (auto& shard : task) { std::memset(shard.addrs[0], 0x6b, 4096); }
                Send(started[1]);
                return UC::Detail::TaskHandle{1};
            }));
            EXPECT_CALL(backend, Wait).WillOnce(testing::Invoke([&](UC::Detail::TaskHandle) {
                Receive(reserved[0]);
                return UC::Status::OK();
            }));
        }
        UC::Detail::BlockId block{};
        auto reservation = early.Reserve(&block, 1);
        Require(bool(reservation), "reserve failed");
        Require(reservation.Value().remainingL2Work == 2, "unexpected initial readiness");
        if (child == 0) { Send(reserved[1]); }
        auto result = early.Prefetch(reservation.Value().handle);
        Require(bool(result), "prefetch failed");
        Require(result.Value().l2ReadShards == (child == 0 ? 0 : 2), "duplicate L2 reader");
        LoadAndRelease(early, manager, config, reservation.Value().handle, block);
        // Each process now has only node-number hints, with no live reservation.
        // Reacquire through its own native attachment; no L2 lookup or read is needed.
        for (size_t repeat = 0; repeat < 2; ++repeat) {
            auto hot = early.Reserve(&block, 1);
            Require(bool(hot), "released hint reserve failed");
            Require(hot.Value().totalHitBlocks == 1 && hot.Value().remainingL2Work == 0,
                    "released hint did not reacquire READY protection");
            LoadAndRelease(early, manager, config, hot.Value().handle, block);
        }
        if (child > 0) {
            int status = 0;
            Require(::waitpid(child, &status, 0) == child, "wait child");
            Require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "child failed");
        }
    };
    if (child == 0) {
        try { run(); }
        catch (...) { ::_exit(1); }
        ::_exit(testing::Test::HasFailure() ? 1 : 0);
    }
    EXPECT_NO_THROW(run());
    ::close(started[0]);
    ::close(started[1]);
    ::close(reserved[0]);
    ::close(reserved[1]);
}
