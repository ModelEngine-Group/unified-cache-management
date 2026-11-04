/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "cache/v2/ctrl_strategy.h"
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <new>
#include <string>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace UC::Cache2 {

struct CtrlStrategyTestAccess {
    static const void* HeaderAddress(const CtrlLayout& layout) { return layout.Hdr(); }
    static size_t RankCount(const CtrlLayout& layout) { return layout.RankCount(); }
    static size_t SlotsPerRank(const CtrlLayout& layout) { return layout.SlotsPerRank(); }
    static size_t SlotSize(const CtrlLayout& layout) { return layout.SlotSize(); }
    static size_t TotalSize(size_t bucketCount, size_t lockCount, size_t slotCount)
    {
        return CtrlLayout::TotalSize(bucketCount, lockCount, slotCount);
    }
    static void Bind(CtrlLayout& layout, void* base, size_t rankCount, size_t slotsPerRank,
                     size_t bucketCount, size_t lockCount)
    {
        layout.Bind(base, rankCount, slotsPerRank, bucketCount, lockCount);
    }
};

namespace {

struct ProcessState {
    std::atomic<size_t> setupCount{0};
    std::atomic<size_t> publishCount{0};
    std::atomic<size_t> controlReady[2]{};
};

struct ParticipantReport {
    int32_t group{-1};
    int32_t rank{-2};
    int32_t setupOk{0};
    int32_t sharedValuesVisible{0};
    uint64_t device{0};
    uint64_t inode{0};
    uint64_t rankCount{0};
    uint64_t slotsPerRank{0};
};

// Identify the mapping containing the public Header view, without accessing private members.
bool ReadMappingIdentity(const void* header, ParticipantReport& report)
{
    std::ifstream maps("/proc/self/maps");
    std::string line;
    auto address = reinterpret_cast<uintptr_t>(header);
    while (std::getline(maps, line)) {
        unsigned long long begin{}, end{}, offset{}, inode{};
        unsigned int major{}, minor{};
        char permissions[5]{};
        auto fields = std::sscanf(line.c_str(), "%llx-%llx %4s %llx %x:%x %llu", &begin, &end,
                                  permissions, &offset, &major, &minor, &inode);
        if (fields != 7 || address < begin || address >= end) { continue; }
        if (permissions[3] != 's' || inode == 0 ||
            line.find("memfd:ucm_cache2_ctrl") == std::string::npos) {
            return false;
        }
        report.device = (static_cast<uint64_t>(major) << 32) | minor;
        report.inode = inode;
        return true;
    }
    return false;
}

bool ReadByte(int fd)
{
    char byte{};
    for (;;) {
        auto bytes = ::read(fd, &byte, sizeof(byte));
        if (bytes == sizeof(byte)) { return true; }
        if (bytes < 0 && errno == EINTR) { continue; }
        return false;
    }
}

bool WriteBytes(int fd, const void* data, size_t size)
{
    for (;;) {
        auto bytes = ::write(fd, data, size);
        if (bytes == static_cast<ssize_t>(size)) { return true; }
        if (bytes < 0 && errno == EINTR) { continue; }
        return false;
    }
}

bool ReadBytes(int fd, void* data, size_t size)
{
    auto* out = static_cast<std::byte*>(data);
    size_t offset = 0;
    while (offset < size) {
        auto bytes = ::read(fd, out + offset, size - offset);
        if (bytes > 0) {
            offset += static_cast<size_t>(bytes);
            continue;
        }
        if (bytes < 0 && errno == EINTR) { continue; }
        return false;
    }
    return true;
}

bool WaitFor(const std::atomic<size_t>& value, size_t expected)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (value.load(std::memory_order_acquire) != expected) {
        if (std::chrono::steady_clock::now() >= deadline) { return false; }
        std::this_thread::yield();
    }
    return true;
}

Config MakeControlConfig(const std::string& uniqueId, int rank)
{
    Config cfg;
    cfg.uniqueId = uniqueId;
    cfg.deviceId = rank;
    cfg.localRankSize = 8;
    cfg.shardSize = 4096;
    cfg.bufferCapacity = cfg.localRankSize * 8 * cfg.shardSize;
    if (rank < 0) { cfg.shardSize = 0; }
    cfg.timeoutMs = 10000;
    return cfg;
}

std::string ControlTestId(const char* name)
{
    return std::string(name) + "_" + std::to_string(::getpid()) + "_" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
}

TEST(Cache2CtrlStrategyTest, SchedulerWithoutShardSizeStartsAfterWorker)
{
    auto id = ControlTestId("scheduler_no_shard");
    auto workerConfig = MakeControlConfig(id, 0);
    auto schedulerConfig = MakeControlConfig(id, -1);
    CtrlStrategy worker;
    CtrlStrategy scheduler;
    ASSERT_TRUE(worker.Setup(workerConfig).Success());
    ASSERT_TRUE(scheduler.Setup(schedulerConfig).Success());
    EXPECT_EQ(schedulerConfig.shardSize, 0);
    auto& layout = scheduler.Layout();
    EXPECT_EQ(CtrlStrategyTestAccess::SlotSize(layout), workerConfig.shardSize);
    EXPECT_EQ(CtrlStrategyTestAccess::RankCount(layout), workerConfig.localRankSize);
    EXPECT_EQ(CtrlStrategyTestAccess::SlotsPerRank(layout), 8);
    CtrlLayout::RankDataDesc desc;
    desc.handle.store(123, std::memory_order_relaxed);
    ASSERT_TRUE(worker.Layout().SetRankDesc(0, desc).Success());
    auto observed = layout.GetRankDesc(0);
    ASSERT_TRUE(observed);
    EXPECT_EQ(observed.Value().handle.load(std::memory_order_relaxed), 123);
}

TEST(Cache2CtrlStrategyTest, SchedulerStillRejectsMismatchedCapacityAndRanks)
{
    auto id = ControlTestId("scheduler_mismatch");
    CtrlStrategy worker;
    ASSERT_TRUE(worker.Setup(MakeControlConfig(id, 0)).Success());
    auto cfg = MakeControlConfig(id, -1);
    cfg.bufferCapacity *= 2;
    CtrlStrategy wrongCapacity;
    EXPECT_EQ(wrongCapacity.Setup(cfg), Status::InvalidParam());
    cfg = MakeControlConfig(id, -1);
    cfg.localRankSize = 4;
    CtrlStrategy wrongRanks;
    EXPECT_EQ(wrongRanks.Setup(cfg), Status::InvalidParam());
}

TEST(Cache2CtrlStrategyTest, WorkerStillRequiresMatchingShardSize)
{
    auto id = ControlTestId("worker_shard");
    auto cfg = MakeControlConfig(id, 0);
    cfg.shardSize = 0;
    CtrlStrategy invalidCreator;
    EXPECT_EQ(invalidCreator.Setup(cfg), Status::InvalidParam());
    CtrlStrategy worker;
    ASSERT_TRUE(worker.Setup(MakeControlConfig(id, 0)).Success());
    cfg.deviceId = 1;
    CtrlStrategy invalidJoiner;
    EXPECT_EQ(invalidJoiner.Setup(cfg), Status::InvalidParam());
    cfg.shardSize = 8192;
    CtrlStrategy mismatchedJoiner;
    EXPECT_EQ(mismatchedJoiner.Setup(cfg), Status::InvalidParam());
}

[[noreturn]] void RunParticipant(int startFd, int releaseFd, int reportFd, ProcessState* state,
                                 int group, int rank, const std::string& uniqueId)
{
    ParticipantReport report;
    report.group = group;
    report.rank = rank;
    if (!ReadByte(startFd)) { ::_exit(2); }
    // Match engine startup: a worker creates each DP control region before its scheduler.
    if (rank < 0 && !WaitFor(state->controlReady[group], 1)) { ::_exit(6); }

    {
        CtrlStrategy strategy;
        auto setup = strategy.Setup(MakeControlConfig(uniqueId, rank));
        report.setupOk = setup.Success();
        if (setup.Success()) {
            if (rank >= 0) { state->controlReady[group].store(1, std::memory_order_release); }
            auto& layout = strategy.Layout();
            if (!ReadMappingIdentity(CtrlStrategyTestAccess::HeaderAddress(layout), report)) {
                report.setupOk = 0;
            }
            report.rankCount = CtrlStrategyTestAccess::RankCount(layout);
            report.slotsPerRank = CtrlStrategyTestAccess::SlotsPerRank(layout);
            state->setupCount.fetch_add(1, std::memory_order_acq_rel);

            if (WaitFor(state->setupCount, 18)) {
                if (rank >= 0) {
                    CtrlLayout::RankDataDesc desc;
                    desc.handle.store(static_cast<size_t>((group + 1) * 100 + rank),
                                      std::memory_order_relaxed);
                    if (layout.SetRankDesc(static_cast<size_t>(rank), desc).Success()) {
                        state->publishCount.fetch_add(1, std::memory_order_acq_rel);
                    }
                }
                if (WaitFor(state->publishCount, 16)) {
                    bool visible = true;
                    for (size_t worker = 0; worker < 8; ++worker) {
                        auto desc = layout.GetRankDesc(worker);
                        auto expected = static_cast<size_t>((group + 1) * 100 + worker);
                        if (!desc ||
                            desc.Value().handle.load(std::memory_order_relaxed) != expected) {
                            visible = false;
                            break;
                        }
                    }
                    report.sharedValuesVisible = visible;
                }
            }
        }

        if (!WriteBytes(reportFd, &report, sizeof(report))) { ::_exit(3); }
        if (!ReadByte(releaseFd)) { ::_exit(4); }
    }
    ::_exit(report.setupOk && report.sharedValuesVisible ? 0 : 5);
}

TEST(Cache2CtrlStrategyProcessTest, TwoDpEightTpShareExactlyTwoControlRegions)
{
    constexpr size_t kGroups{2};
    constexpr size_t kParticipantsPerGroup{9};
    constexpr size_t kParticipants{kGroups * kParticipantsPerGroup};
    int startPipe[2]{};
    int releasePipe[2]{};
    int reportPipe[2]{};
    ASSERT_EQ(::pipe(startPipe), 0);
    ASSERT_EQ(::pipe(releasePipe), 0);
    ASSERT_EQ(::pipe(reportPipe), 0);

    auto* state = static_cast<ProcessState*>(::mmap(
        nullptr, sizeof(ProcessState), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    ASSERT_NE(state, MAP_FAILED);
    ::new (state) ProcessState();

    auto nonce = std::to_string(::getpid()) + "_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::array<std::string, kGroups> uniqueIds = {"cache2_dp0_" + nonce, "cache2_dp1_" + nonce};
    std::vector<pid_t> children;
    children.reserve(kParticipants);
    for (size_t group = 0; group < kGroups; ++group) {
        for (int rank = -1; rank < 8; ++rank) {
            auto child = ::fork();
            ASSERT_GE(child, 0);
            if (child == 0) {
                ::close(startPipe[1]);
                ::close(releasePipe[1]);
                ::close(reportPipe[0]);
                RunParticipant(startPipe[0], releasePipe[0], reportPipe[1], state,
                               static_cast<int>(group), rank, uniqueIds[group]);
            }
            children.push_back(child);
        }
    }
    ::close(startPipe[0]);
    ::close(releasePipe[0]);
    ::close(reportPipe[1]);

    std::array<char, kParticipants> signals{};
    ASSERT_TRUE(WriteBytes(startPipe[1], signals.data(), signals.size()));
    ::close(startPipe[1]);

    std::vector<ParticipantReport> reports(kParticipants);
    bool reportsRead = true;
    for (auto& report : reports) {
        if (!ReadBytes(reportPipe[0], &report, sizeof(report))) {
            reportsRead = false;
            break;
        }
    }
    ::close(reportPipe[0]);

    EXPECT_TRUE(WriteBytes(releasePipe[1], signals.data(), signals.size()));
    ::close(releasePipe[1]);
    EXPECT_TRUE(reportsRead);

    for (auto child : children) {
        int status = 0;
        EXPECT_EQ(::waitpid(child, &status, 0), child);
        EXPECT_TRUE(WIFEXITED(status));
        if (WIFEXITED(status)) { EXPECT_EQ(WEXITSTATUS(status), 0); }
    }

    std::array<uint64_t, kGroups> groupDevice{};
    std::array<uint64_t, kGroups> groupInode{};
    std::array<size_t, kGroups> participants{};
    std::array<bool, kGroups> schedulers{};
    std::array<std::array<bool, 8>, kGroups> workers{};
    for (const auto& report : reports) {
        ASSERT_GE(report.group, 0);
        ASSERT_LT(report.group, static_cast<int32_t>(kGroups));
        auto group = static_cast<size_t>(report.group);
        EXPECT_TRUE(report.setupOk);
        EXPECT_TRUE(report.sharedValuesVisible);
        EXPECT_EQ(report.rankCount, 8);
        EXPECT_EQ(report.slotsPerRank, 8);
        EXPECT_NE(report.inode, 0);
        if (participants[group] == 0) {
            groupDevice[group] = report.device;
            groupInode[group] = report.inode;
        } else {
            EXPECT_EQ(report.device, groupDevice[group]);
            EXPECT_EQ(report.inode, groupInode[group]);
        }
        ++participants[group];
        if (report.rank == -1) {
            EXPECT_FALSE(schedulers[group]);
            schedulers[group] = true;
        } else {
            ASSERT_GE(report.rank, 0);
            ASSERT_LT(report.rank, 8);
            auto rank = static_cast<size_t>(report.rank);
            EXPECT_FALSE(workers[group][rank]);
            workers[group][rank] = true;
        }
    }
    EXPECT_EQ(participants[0], kParticipantsPerGroup);
    EXPECT_EQ(participants[1], kParticipantsPerGroup);
    EXPECT_TRUE(schedulers[0]);
    EXPECT_TRUE(schedulers[1]);
    for (size_t group = 0; group < kGroups; ++group) {
        for (size_t rank = 0; rank < 8; ++rank) { EXPECT_TRUE(workers[group][rank]); }
    }
    EXPECT_TRUE(groupDevice[0] != groupDevice[1] || groupInode[0] != groupInode[1]);
    std::cout << "DP0 control=(dev " << groupDevice[0] << ", inode " << groupInode[0]
              << "), participants=9, rankCount=8; DP1 control=(dev " << groupDevice[1] << ", inode "
              << groupInode[1] << "), participants=9, rankCount=8; distinct controls=2"
              << std::endl;

    state->~ProcessState();
    EXPECT_EQ(::munmap(state, sizeof(ProcessState)), 0);
}

TEST(Cache2CtrlLayoutProcessTest, BucketStripeLockSerializesAcrossProcesses)
{
    constexpr size_t kRanks{1};
    constexpr size_t kSlotsPerRank{1};
    constexpr size_t kBuckets{8};
    constexpr size_t kLocks{4};
    auto bytes = CtrlStrategyTestAccess::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* memory =
        ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(memory, MAP_FAILED);
    CtrlLayout layout;
    CtrlStrategyTestAccess::Bind(layout, memory, kRanks, kSlotsPerRank, kBuckets, kLocks);
    layout.InitHeader(4096);
    layout.InitSlotRange(0);
    ASSERT_EQ(layout.LockOf(0), layout.LockOf(4));

    int commandPipe[2]{};
    int resultPipe[2]{};
    ASSERT_EQ(::pipe(commandPipe), 0);
    ASSERT_EQ(::pipe(resultPipe), 0);
    layout.LockOf(0)->Lock();
    auto child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        ::close(commandPipe[1]);
        ::close(resultPipe[0]);
        uint8_t first = layout.LockOf(4)->TryLock();
        if (first != 0) { layout.LockOf(4)->Unlock(); }
        if (!WriteBytes(resultPipe[1], &first, sizeof(first)) || !ReadByte(commandPipe[0])) {
            ::_exit(2);
        }
        uint8_t second = layout.LockOf(4)->TryLock();
        if (second != 0) { layout.LockOf(4)->Unlock(); }
        if (!WriteBytes(resultPipe[1], &second, sizeof(second))) { ::_exit(3); }
        ::_exit(first == 0 && second == 1 ? 0 : 4);
    }
    ::close(commandPipe[0]);
    ::close(resultPipe[1]);
    uint8_t first{};
    ASSERT_TRUE(ReadBytes(resultPipe[0], &first, sizeof(first)));
    EXPECT_EQ(first, 0);
    layout.LockOf(0)->Unlock();
    ASSERT_TRUE(WriteBytes(commandPipe[1], "x", 1));
    uint8_t second{};
    ASSERT_TRUE(ReadBytes(resultPipe[0], &second, sizeof(second)));
    EXPECT_EQ(second, 1);
    ::close(commandPipe[1]);
    ::close(resultPipe[0]);
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
    EXPECT_EQ(::munmap(memory, bytes), 0);
}

}  // namespace
}  // namespace UC::Cache2
