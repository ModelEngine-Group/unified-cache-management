// Native shared L1 and load queues in two processes; backend/HBM are simulated.
#include <array>
#include <cstring>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#include <gtest/gtest.h>
#include "cache/cc/load_queue.h"
#include "cache/cc/dump_queue.h"
#include "cache/cc/reserved_prefetch.h"
#include "detail/mock_store.h"
#include "detail/random.h"

namespace {
class Backend : public testing::NiceMock<UC::Test::Detail::MockStore> {
public:
    Backend() { ON_CALL(*this, Check).WillByDefault(testing::Invoke([](auto) { return true; })); }

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

namespace {
UC::CacheStore::Config SharedConfig()
{
    UC::CacheStore::Config config;
    config.uniqueId = UC::Test::Detail::Random().RandomString(16);
    config.deviceId = 0;
    config.tensorSizes = {4096};
    config.shardSize = 4096;
    config.blockSize = 8192;
    config.bufferCapacity = 4 << 20;
    config.loadExclusiveBufferNumber = 16;
    config.shareBufferEnable = true;
    return config;
}

void SendText(int fd, const std::string& value)
{
    const auto size = value.size();
    Require(::write(fd, &size, sizeof(size)) == sizeof(size), "send length");
    Require(::write(fd, value.data(), size) == static_cast<ssize_t>(size), "send identity");
}

std::string ReceiveText(int fd)
{
    size_t size = 0;
    Require(::read(fd, &size, sizeof(size)) == sizeof(size) && size < 1024, "receive length");
    std::string value(size, '\0');
    Require(::read(fd, value.data(), size) == static_cast<ssize_t>(size), "receive identity");
    return value;
}

void NativeLoad(UC::CacheStore::BufferManager& manager,
                const UC::CacheStore::Config& config, const UC::Detail::BlockId& block)
{
    std::array<unsigned char, 8192> destination{};
    UC::HashSet<UC::Detail::TaskHandle> failures;
    UC::CacheStore::LoadQueue queue;
    Require(queue.Setup(config, &failures, manager.GetTransBuffer()).Success(), "native load setup");
    for (size_t layer = 0; layer < 2; ++layer) {
        UC::Detail::TaskDesc desc{{block, layer, {destination.data() + layer * 4096}}};
        auto task = std::make_shared<UC::CacheStore::TransTask>(
            UC::CacheStore::TransTask::Type::LOAD, desc);
        Require(task->loadHandles.empty(), "consumer must use native Get");
        auto waiter = std::make_shared<UC::Latch>();
        queue.Submit(task, waiter);
        Require(waiter->WaitForDuration(5000), "native simulated H2D timeout");
        Require(!failures.Contains(task->id), "native simulated H2D failed");
    }
    for (auto byte : destination) { Require(byte == 0x6b, "consumer HBM target differs"); }
}
}

TEST(UCCacheEarlyProcessTest, OneOwnerTwoNativeConsumersDelayedUnderEvictionPressure)
{
    auto config = SharedConfig();
    std::array<std::array<int, 2>, 2> start{}, done{};
    for (size_t i = 0; i < 2; ++i) {
        ASSERT_EQ(::pipe(start[i].data()), 0);
        ASSERT_EQ(::pipe(done[i].data()), 0);
    }
    std::array<pid_t, 2> children{};
    int consumer = -1;
    for (size_t i = 0; i < 2; ++i) {
        children[i] = ::fork();  // no native threads or shared mappings before fork
        ASSERT_GE(children[i], 0);
        if (children[i] == 0) { consumer = i; break; }
    }
    auto run = [&] {
        Backend backend;
        config.storeBackend = &backend;
        UC::Detail::BlockId block{};
        if (consumer >= 0) {
            Receive(start[consumer][0]);
            const auto identity = ReceiveText(start[consumer][0]);
            config.deviceId = consumer + 1;
            UC::CacheStore::BufferManager manager;
            Require(manager.Setup(config).Success(), "consumer attach");
            auto domain = UC::CacheStore::EarlyBufferPins::DescribeDomain(*manager.GetTransBuffer());
            Require(!domain.backing.empty() && domain.backing == identity, "different native backing");
            EXPECT_CALL(backend, LookupOnPrefix).Times(0);
            EXPECT_CALL(backend, Load).Times(0);
            EXPECT_CALL(backend, Wait).Times(0);
            NativeLoad(manager, config, block);  // no reserve/lease or cross-process handle
            Send(done[consumer][1]);
            Receive(start[consumer][0]);  // keep attachment alive until owner's validation ends
            return;
        }
        UC::CacheStore::BufferManager manager;
        Require(manager.Setup(config).Success(), "owner setup");
        auto& buffer = *manager.GetTransBuffer();
        auto domain = UC::CacheStore::EarlyBufferPins::DescribeDomain(buffer);
        Require(!domain.backing.empty() && domain.nodes > domain.reservable, "shared budget/headroom");
        UC::CacheStore::ReservedPrefetch early(&manager, &backend, config.blockSize,
                                               config.shardSize, config.timeoutMs);
        EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(ssize_t{0}));
        EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([](UC::Detail::TaskDesc task) {
            for (auto& shard : task) { std::memset(shard.addrs[0], 0x6b, 4096); }
            return UC::Detail::TaskHandle{1};
        }));
        EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::OK()));
        auto reservation = early.Reserve(&block, 1);
        Require(bool(reservation), "owner reserve");
        auto handle = reservation.Value().handle;
        Require(bool(early.Prefetch(handle)), "owner prefetch");
        Send(start[0][1]);
        SendText(start[0][1], domain.backing);
        Receive(done[0][0]);
        // Consumer 1 has not acquired any references. Evict many other shards.
        for (size_t i = 1; i < domain.nodes * 2; ++i) {
            UC::Detail::BlockId other{};
            std::memcpy(&other, &i, sizeof(i));
            auto pressure = buffer.TryGet(other, 0);
            if (pressure) { pressure.MarkReady(); }
        }
        Require(reservation.Value().batch->ProtectedBytes() >= 8192, "owner released before late consumer");
        Send(start[1][1]);
        SendText(start[1][1], domain.backing);
        Receive(done[1][0]);
        // Conservative protocol: both native queues have completed all H2D.
        early.Release(handle);
        Require(reservation.Value().batch->ProtectedBytes() == 0, "protection not returned");
        for (size_t i = 0; i < 2; ++i) { Send(start[i][1]); }
        for (auto child : children) {
            int status = 0;
            Require(::waitpid(child, &status, 0) == child, "wait consumer");
            Require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "consumer failed");
        }
    };
    if (consumer >= 0) {
        try { run(); } catch (...) { ::_exit(1); }
        ::_exit(testing::Test::HasFailure() ? 1 : 0);
    }
    EXPECT_NO_THROW(run());
    for (size_t i = 0; i < 2; ++i) {
        for (auto fd : start[i]) { ::close(fd); }
        for (auto fd : done[i]) { ::close(fd); }
    }
}

TEST(UCCacheEarlyProcessTest, ProcessDeathDoesNotRecoverNativeSharedReferences)
{
    auto config = SharedConfig();
    int start[2], acquired[2];
    ASSERT_EQ(::pipe(start), 0);
    ASSERT_EQ(::pipe(acquired), 0);
    const auto child = ::fork();
    ASSERT_GE(child, 0);
    UC::Detail::BlockId block{};
    if (child == 0) {
        try {
            Receive(start[0]);
            UC::CacheStore::TransBuffer buffer;
            Require(buffer.Setup(config).Success(), "child attach");
            auto held = buffer.Get(block, 0);
            Require(bool(held) && held.Ready(), "child acquire native reference");
            Send(acquired[1]);
            ::_exit(0);  // no RAII: models unexpected process death outside a lock
        } catch (...) { ::_exit(1); }
    }
    UC::CacheStore::TransBuffer buffer;
    ASSERT_TRUE(buffer.Setup(config).Success());
    auto held = buffer.Get(block, 0);
    ASSERT_TRUE(held);
    held.MarkReady();
    Send(start[1]);
    Receive(acquired[0]);
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    held = {};
    auto domain = UC::CacheStore::EarlyBufferPins::DescribeDomain(buffer);
    std::vector<UC::CacheStore::TransBuffer::Handle> occupied;
    for (size_t i = 1; i <= domain.reservable; ++i) {
        UC::Detail::BlockId other{};
        std::memcpy(&other, &i, sizeof(i));
        auto pin = buffer.TryGet(other, 0);
        if (!pin) { break; }
        pin.MarkReady();
        occupied.push_back(std::move(pin));
    }
    EXPECT_EQ(occupied.size(), domain.reservable - 1);  // dead reference is NOT recovered
    for (auto fd : start) { ::close(fd); }
    for (auto fd : acquired) { ::close(fd); }
    // This test owns the unique backing; normal buffer destruction unlinks it.
}


TEST(UCCacheEarlyProcessTest, FullOwnerCreditLeavesTwoLoadsAndAggregateDumpRoomToDrain)
{
    auto config = SharedConfig();
    Backend backend;
    config.storeBackend = &backend;
    UC::CacheStore::BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    auto* buffer = manager.GetTransBuffer();
    const auto domain = UC::CacheStore::EarlyBufferPins::DescribeDomain(*buffer);
    constexpr size_t loadTask = 8, dumpTask = 8;
    ASSERT_GE(domain.nodes, 2 * loadTask + dumpTask);
    ASSERT_GE(domain.reservable, dumpTask);
    const auto slots = std::min(domain.reservable - dumpTask,
                                domain.nodes - 2 * loadTask - dumpTask);
    auto makeBlock = [](uint64_t value) {
        UC::Detail::BlockId block{};
        std::memcpy(&block, &value, sizeof(value));
        return block;
    };
    std::vector<UC::Detail::BlockId> blocks;
    for (size_t i = 0; i < slots / 2; ++i) { blocks.push_back(makeBlock(i)); }
    EXPECT_CALL(backend, LookupOnPrefix).WillRepeatedly(testing::Invoke(
        [](const UC::Detail::BlockId*, size_t count) { return ssize_t(count - 1); }));
    EXPECT_CALL(backend, Load).WillRepeatedly(testing::Invoke(
        [](UC::Detail::TaskDesc task) {
            for (auto& shard : task) { std::memset(shard.addrs[0], 0x6b, 4096); }
            return UC::Detail::TaskHandle{1};
        }));
    EXPECT_CALL(backend, Wait).WillRepeatedly(testing::Return(UC::Status::OK()));
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, config.blockSize,
                                           config.shardSize, config.timeoutMs);
    auto reservation = early.Reserve(blocks.data(), blocks.size());
    ASSERT_TRUE(reservation);
    ASSERT_TRUE(early.Prefetch(reservation.Value().handle));
    UC::HashSet<UC::Detail::TaskHandle> failures;
    UC::CacheStore::LoadQueue loads[2];
    UC::CacheStore::DumpQueue dumps;
    ASSERT_TRUE(dumps.Setup(config, &failures, buffer).Success());
    for (auto& load : loads) { ASSERT_TRUE(load.Setup(config, &failures, buffer).Success()); }
    std::array<std::array<unsigned char, loadTask * 4096>, 2> targets{};
    std::array<unsigned char, dumpTask * 4096> source{};
    source.fill(0x3c);
    auto persisted = std::make_shared<UC::Latch>();
    EXPECT_CALL(backend, Dump).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc task) {
        EXPECT_EQ(task.size(), dumpTask);
        for (auto& shard : task) {
            EXPECT_EQ(static_cast<unsigned char*>(shard.addrs[0])[0], 0x3c);
        }
        persisted->Done();
        return UC::Detail::TaskHandle{2};
    }));
    std::array<std::shared_ptr<UC::Latch>, 3> waiters;
    std::array<std::shared_ptr<UC::CacheStore::TransTask>, 3> tasks;
    for (size_t q = 0; q < 3; ++q) {
        UC::Detail::TaskDesc desc;
        for (size_t i = 0; i < loadTask; ++i) {
            void* address = q < 2 ? targets[q].data() + i * 4096 : source.data() + i * 4096;
            desc.push_back({makeBlock(100000 + q * loadTask + i), 0, {address}});
        }
        tasks[q] = std::make_shared<UC::CacheStore::TransTask>(q < 2 ?
            UC::CacheStore::TransTask::Type::LOAD : UC::CacheStore::TransTask::Type::DUMP, desc);
        waiters[q] = std::make_shared<UC::Latch>();
        if (q < 2) { loads[q].Submit(tasks[q], waiters[q]); }
        else { dumps.Submit(tasks[q], waiters[q]); }
    }
    for (size_t q = 0; q < 3; ++q) {
        ASSERT_TRUE(waiters[q]->WaitForDuration(5000));
        ASSERT_FALSE(failures.Contains(tasks[q]->id));
    }
    ASSERT_TRUE(persisted->WaitForDuration(5000));
    for (auto& target : targets) {
        for (auto byte : target) { ASSERT_EQ(byte, 0x6b); }
    }
    // Every reserved shard remains protected throughout native Get/D2H/H2D.
    auto handles = early.LoadHandles([&] {
        UC::Detail::TaskDesc desc{{blocks[0], 0, {targets[0].data()}}};
        desc.loadReservations = {reservation.Value().handle};
        return desc;
    }());
    ASSERT_EQ(handles.size(), 1);
    ASSERT_EQ(static_cast<unsigned char*>(handles[0].Data())[0], 0x6b);
    early.Release(reservation.Value().handle);
}
