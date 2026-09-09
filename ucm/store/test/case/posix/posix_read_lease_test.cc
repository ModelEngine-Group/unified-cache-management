#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include "detail/path_base.h"
#include "posix/cc/read_lease.h"
#include "posix/cc/shard_gc.h"

namespace {
using namespace UC;
using namespace UC::PosixStore;
using Detail::BlockId;
using namespace std::chrono_literals;
class UCPosixReadLeaseTest : public UC::Test::Detail::PathBase {};

Config LayoutConfig(const std::string& path)
{
    Config config;
    config.storageBackends = {path};
    config.dataDirShardBytes = 0;
    config.acquireRemoveGuard = PosixReadLease::AcquireRemoveGuard;
    return config;
}

size_t OpenFiles()
{
    return std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                         std::filesystem::directory_iterator{});
}

TEST_F(UCPosixReadLeaseTest, PartialFailureAndExceptionReleaseAllAcquiredFiles)
{
    auto config = LayoutConfig(Path());
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), Status::OK());
    BlockId blocks[2]{};
    blocks[1][0] = std::byte{1};
    auto writer = PosixReadLease::AcquireRemoveGuardForLayout(layout, blocks[1]);
    ASSERT_TRUE(writer);
    const auto before = OpenFiles();
    for (size_t i = 0; i < 32; ++i) {
        auto failed = PosixReadLease::Acquire(layout, blocks, 2);
        ASSERT_FALSE(failed);
        EXPECT_EQ(failed.Error(), Status::Retry());
        auto released = PosixReadLease::AcquireRemoveGuardForLayout(layout, blocks[0]);
        ASSERT_TRUE(released);  // The first lock of the failed batch is gone.
    }
    EXPECT_EQ(OpenFiles(), before);
    writer.Value().reset();
    try {
        auto lease = PosixReadLease::Acquire(layout, blocks, 2);
        ASSERT_TRUE(lease);
        throw std::runtime_error("consumer failure");
    } catch (const std::runtime_error&) {}
    EXPECT_EQ(OpenFiles(), before - 1);
    for (auto& block : blocks) {
        auto released = PosixReadLease::AcquireRemoveGuardForLayout(layout, block);
        EXPECT_TRUE(released);
    }
    struct stat st {};
    ASSERT_EQ(::stat((Path() + "/.ucm_read_leases").c_str(), &st), 0);
    EXPECT_EQ(st.st_mode & 0777, 0600);
}

TEST_F(UCPosixReadLeaseTest, PartialOpenFailureUsesNativeErrorAndClosesEarlierBackend)
{
    auto other = Path() + "/other";
    std::filesystem::create_directories(other + "/data");
    auto config = LayoutConfig(Path());
    config.storageBackends.push_back(other);
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), Status::OK());
    BlockId blocks[2]{};
    bool found[2]{};
    for (unsigned i = 0; i < 256 && !(found[0] && found[1]); ++i) {
        BlockId key{};
        key[0] = std::byte(i);
        auto index = layout.DataFilePath(key, false).rfind(other + "/", 0) == 0 ? 1 : 0;
        blocks[index] = key;
        found[index] = true;
    }
    ASSERT_TRUE(found[0] && found[1]);
    std::filesystem::create_directory(other + "/.ucm_read_leases");
    const auto before = OpenFiles();
    auto failed = PosixReadLease::Acquire(layout, blocks, 2);
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.Error(), Status::OsApiError());
    EXPECT_EQ(OpenFiles(), before);
    auto first = PosixReadLease::AcquireRemoveGuardForLayout(layout, blocks[0]);
    EXPECT_TRUE(first);
}

std::string guardedPath;
bool guardSawUnlink = false;
bool guardStillExcludedReader = false;
Expected<std::shared_ptr<void>> ObserveRemoveGuard(const std::string& backend, const BlockId& block)
{
    auto acquired = PosixReadLease::AcquireRemoveGuard(backend, block);
    if (!acquired) { return acquired.Error(); }
    auto held = std::move(acquired.Value());
    auto address = held.get();
    return std::shared_ptr<void>(address, [held, backend, block](void*) {
        guardSawUnlink = !std::filesystem::exists(guardedPath);
        PosixReadLease reader;
        guardStillExcludedReader = reader.Lock(backend, block) == Status::Retry();
    });
}

TEST_F(UCPosixReadLeaseTest, GuardHoldsExclusiveLockThroughActualUnlink)
{
    auto config = LayoutConfig(Path());
    config.acquireRemoveGuard = ObserveRemoveGuard;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), Status::OK());
    BlockId block{};
    guardedPath = layout.DataFilePath(block, false);
    std::ofstream(guardedPath) << "data";
    guardSawUnlink = guardStillExcludedReader = false;
    ASSERT_EQ(layout.RemoveFile(block), Status::OK());
    EXPECT_TRUE(guardSawUnlink);
    EXPECT_TRUE(guardStillExcludedReader);
    auto released = PosixReadLease::Acquire(layout, &block, 1);
    EXPECT_TRUE(released);
}

struct Observation {
    std::atomic<unsigned> attempts{0};
    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};
};
Observation* observation;
Expected<std::shared_ptr<void>> CountRemoveGuard(const std::string& backend, const BlockId& block)
{
    observation->attempts.fetch_add(1);
    return PosixReadLease::AcquireRemoveGuard(backend, block);
}

template <typename Predicate> bool Await(Predicate done, std::chrono::milliseconds timeout = 5s)
{
    const auto until = std::chrono::steady_clock::now() + timeout;
    while (!done() && std::chrono::steady_clock::now() < until) { std::this_thread::sleep_for(10ms); }
    return done();
}

struct Child {
    pid_t pid{-1};
    ~Child() { if (pid > 0) { ::kill(pid, SIGKILL); ::waitpid(pid, nullptr, 0); } }
};

void ExerciseGc(const std::string& path, bool precise)
{
    auto config = LayoutConfig(path);
    config.acquireRemoveGuard = CountRemoveGuard;
    config.blockSize = 256ULL << 20;  // Four logical files fill this test's 1 GiB capacity.
    config.posixCapacityGb = 1;
    config.posixGcConcurrency = 1;
    config.posixGcPreciseMode = precise;
    config.posixGcRecyclePercent = 0.5;
    config.posixGcCandidateExtraPercent = 0;  // Three-file fixture must select at least one victim.
    config.posixGcTriggerThresholdRatio = 0.5;
    config.posixGcShardSampleRatio = 1.0;
    config.posixGcCheckIntervalSec = 1;  // Exercise the native wait without a long test.
    config.posixGcHeartbeatIntervalSec = 1;
    config.posixGcStaleThresholdSec = 60;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), Status::OK());
    BlockId blocks[3]{};
    for (size_t i = 0; i < 3; ++i) {
        blocks[i][0] = std::byte(i);
        std::ofstream(layout.DataFilePath(blocks[i], false)) << "data";
    }
    void* memory = ::mmap(nullptr, sizeof(Observation), PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(memory, MAP_FAILED);
    observation = new (memory) Observation;
    std::unique_ptr<Observation, void(*)(Observation*)> mapping(observation, [](Observation* p) {
        p->~Observation();
        ::munmap(p, sizeof(Observation));
    });
    Child child;
    child.pid = ::fork();  // No reader fds or worker threads exist at fork.
    ASSERT_GE(child.pid, 0);
    if (child.pid == 0) {
        if (!Await([] { return observation->start.load(); })) { _exit(10); }
        {
            ShardGarbageCollector gc;
            if (gc.Setup(&layout, config).Failure()) { _exit(11); }
            if (!Await([] { return observation->stop.load(); }, 15s)) { _exit(12); }
        }  // Native StopBackgroundCheck / pool / owner lease destruction.
        _exit(0);
    }
    auto first = PosixReadLease::Acquire(layout, blocks, 3);
    auto second = PosixReadLease::Acquire(layout, blocks, 3);
    ASSERT_TRUE(first && second);
    struct stat original {};
    ASSERT_EQ(::stat((path + "/.ucm_read_leases").c_str(), &original), 0);
    observation->start.store(true);
    ASSERT_TRUE(Await([] { return observation->attempts.load() > 0; }));
    std::this_thread::sleep_for(100ms);  // Let the current native GC round settle.
    const auto attempts = observation->attempts.load();
    std::this_thread::sleep_for(300ms);
    EXPECT_EQ(observation->attempts.load(), attempts);  // No repeated drain/spin.
    EXPECT_EQ(layout.CountFilesInShard("data"), 3);
    first.Value().reset();
    ASSERT_TRUE(Await([attempts] { return observation->attempts.load() > attempts; }));
    EXPECT_EQ(layout.CountFilesInShard("data"), 3);  // Second OFD still protects all files.
    second.Value().reset();
    ASSERT_TRUE(Await([&] { return layout.CountFilesInShard("data") < 3; }));
    observation->stop.store(true);
    int status = 0;
    ASSERT_EQ(::waitpid(child.pid, &status, 0), child.pid);
    child.pid = -1;
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    struct stat after {};
    ASSERT_EQ(::stat((path + "/.ucm_read_leases").c_str(), &after), 0);
    EXPECT_EQ(original.st_ino, after.st_ino);
    std::cout << "gc_precise=" << precise << " attempts=" << observation->attempts.load()
              << " remaining_files=" << layout.CountFilesInShard("data") << std::endl;
}

TEST_F(UCPosixReadLeaseTest, PreciseGcOwnerWaitsThenReclaims) { ExerciseGc(Path(), true); }
TEST_F(UCPosixReadLeaseTest, ShardGcOwnerWaitsThenReclaims) { ExerciseGc(Path(), false); }
}  // namespace
