#include "bench_runner.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include "buffer_allocator.h"
#include "key_value_generator.h"
#include "kv_test_config_helpers.h"
#include "payload_buffer_runtime.h"

namespace kv::bench {

std::string FormatMiBPerSec(double bytesPerSec)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << (bytesPerSec / (1024.0 * 1024.0));
    return stream.str();
}

namespace {

constexpr int kExitInvalidArgument = 1;

struct BenchBufferSlot {
    BufferSet buffers;
};

struct OperationOutcome {
    Status status;
    double latencyUs{0.0};
    std::size_t entryCount{0};
    std::uint64_t bytes{0};
};

using BenchBufferPool = std::vector<BenchBufferSlot>;

class BenchExecutor {
public:
    explicit BenchExecutor(std::size_t workerCount)
    {
        workers_.reserve(workerCount);
        try {
            for (std::size_t index = 0; index < workerCount; ++index) {
                workers_.emplace_back(&BenchExecutor::WorkerLoop, this);
            }
        } catch (...) {
            Shutdown();
            throw;
        }
    }

    ~BenchExecutor() { Shutdown(); }

    BenchExecutor(const BenchExecutor&) = delete;
    BenchExecutor& operator=(const BenchExecutor&) = delete;

    std::future<OperationOutcome> Submit(std::function<OperationOutcome()> handler)
    {
        std::packaged_task<OperationOutcome()> task(std::move(handler));
        auto future = task.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) { throw std::runtime_error("bench executor is stopping"); }
            tasks_.emplace_back(std::move(task));
        }
        workReady_.notify_one();
        return future;
    }

private:
    void Shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        workReady_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) { worker.join(); }
        }
        workers_.clear();
    }

    void WorkerLoop()
    {
        while (true) {
            std::packaged_task<OperationOutcome()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                workReady_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) { return; }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable workReady_;
    std::deque<std::packaged_task<OperationOutcome()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_{false};
};

std::uint64_t BenchEntryCount(const BenchConfig& bench, BenchOpType op)
{
    if (op == BenchOpType::BATCH_STORE || op == BenchOpType::BATCH_RETRIEVE ||
        op == BenchOpType::MIX) {
        return std::max<std::uint32_t>(bench.batchSize, 1);
    }
    return 1;
}

double PercentileUs(const std::vector<double>& sortedLatenciesUs, double percentile)
{
    if (sortedLatenciesUs.empty()) { return 0.0; }
    const double rank =
        std::ceil((percentile / 100.0) * static_cast<double>(sortedLatenciesUs.size()));
    const auto index = static_cast<std::size_t>(
        std::min<double>(std::max<double>(rank, 1.0), sortedLatenciesUs.size()) - 1.0);
    return sortedLatenciesUs[index];
}

BenchLatencyStats BuildLatencyStats(const std::vector<double>& latenciesUs)
{
    BenchLatencyStats stats;
    if (latenciesUs.empty()) { return stats; }

    auto sortedLatenciesUs = latenciesUs;
    std::sort(sortedLatenciesUs.begin(), sortedLatenciesUs.end());
    stats.minUs = sortedLatenciesUs.front();
    stats.maxUs = sortedLatenciesUs.back();
    stats.avgUs = std::accumulate(sortedLatenciesUs.begin(), sortedLatenciesUs.end(), 0.0) /
                  static_cast<double>(sortedLatenciesUs.size());
    stats.p99_9Us = PercentileUs(sortedLatenciesUs, 99.9);
    stats.p99_99Us = PercentileUs(sortedLatenciesUs, 99.99);
    stats.p99_999Us = PercentileUs(sortedLatenciesUs, 99.999);
    return stats;
}

std::string FormatUs(double latencyUs)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(1) << latencyUs;
    return stream.str();
}

void PrintProgressSample(const BenchRealtimeSample& sample, std::uint64_t operationsPerSec)
{
    std::cout << '[' << sample.timestampSec << "s] ops=" << operationsPerSec
              << " entries/s=" << static_cast<std::uint64_t>(sample.iops)
              << " bw=" << FormatMiBPerSec(sample.bandwidthBytesPerSec)
              << "MiB/s avg=" << FormatUs(sample.avgLatencyUs) << "us"
              << " err=" << sample.errorCount << '\n';
}

Status CheckBenchMemoryLimit(std::uint64_t entryCount, std::uint64_t ioSize,
                             std::uint64_t memoryMaxBytes)
{
    if (memoryMaxBytes == 0) {
        return Status::Error(kExitInvalidArgument,
                             "limits.memory_max_bytes must be greater than zero");
    }
    if (entryCount == 0 || ioSize == 0) { return Status::Success(); }
    if (entryCount > std::numeric_limits<std::uint64_t>::max() / ioSize) {
        return Status::Error(kExitInvalidArgument, "bench buffer pool bytes overflow uint64");
    }

    const auto requiredBytes = entryCount * ioSize;
    if (requiredBytes > memoryMaxBytes) {
        return Status::Error(kExitInvalidArgument,
                             "bench buffer pool bytes exceed limits.memory_max_bytes: required=" +
                                 std::to_string(requiredBytes) +
                                 ", limit=" + std::to_string(memoryMaxBytes));
    }
    return Status::Success();
}

Status ValidateBenchBufferConfig(const BenchConfig& bench, std::uint64_t poolEntryCount,
                                 std::uint64_t memoryMaxBytes)
{
    if (bench.ioSize == 0) {
        return Status::Error(kExitInvalidArgument, "bench io_size must be greater than zero");
    }
    if (poolEntryCount > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Status::Error(kExitInvalidArgument,
                             "bench buffer pool entry count exceeds addressable memory");
    }
    if (bench.ioSize > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Status::Error(kExitInvalidArgument, "bench io_size exceeds addressable memory");
    }

    return CheckBenchMemoryLimit(poolEntryCount, bench.ioSize, memoryMaxBytes);
}

void FillStoreValue(std::vector<std::uint8_t>& value, std::uint64_t valueIndex, std::uint64_t seed)
{
    for (std::size_t byteIndex = 0; byteIndex < value.size(); ++byteIndex) {
        value[byteIndex] = static_cast<std::uint8_t>((valueIndex + byteIndex + seed) & 0xFF);
    }
}

Status SyncBenchDeviceBuffers(const KvTestConfig& config, BenchBufferSlot& slot,
                              std::size_t entryCount)
{
    auto status = MaybeSetUpPayloadThread(config);
    if (!status.Ok()) { return status; }

    auto& buffers = slot.buffers;
    if (buffers.ownedBuffers.size() < entryCount || buffers.deviceBuffers.empty() ||
        buffers.deviceBufferOffsets.size() < entryCount) {
        return Status::Error(kExitInvalidArgument, "device payload bench buffer count mismatch");
    }
    const auto baseAddr = reinterpret_cast<std::uintptr_t>(buffers.deviceBuffers.front().get());
    for (std::size_t index = 0; index < entryCount; ++index) {
        status = CopyHostToDevice(buffers.ownedBuffers[index],
                                  baseAddr + buffers.deviceBufferOffsets[index],
                                  "bench index=" + std::to_string(index));
        if (!status.Ok()) { return status; }
    }
    return Status::Success();
}

Status BuildBenchBufferPool(const KvTestConfig& config, bool useDeviceBuffers,
                            std::uint64_t entryCountPerOperation, BenchBufferPool& pool)
{
    const auto& bench = config.bench;
    const auto allocationPolicy = AllocationPolicyForConfig(config);
    if (useDeviceBuffers) {
        auto status = MaybeSetUpPayloadThread(config);
        if (!status.Ok()) { return status; }
    }

    pool.resize(bench.concurrency);
    for (std::size_t slotIndex = 0; slotIndex < pool.size(); ++slotIndex) {
        auto& buffers = pool[slotIndex].buffers;
        buffers.ownedBuffers.reserve(static_cast<std::size_t>(entryCountPerOperation));
        buffers.deviceBufferOffsets.reserve(static_cast<std::size_t>(entryCountPerOperation));
        std::size_t deviceBufferSize = 0;
        for (std::uint64_t index = 0; index < entryCountPerOperation; ++index) {
            auto& value = buffers.ownedBuffers.emplace_back(static_cast<std::size_t>(bench.ioSize));
            FillStoreValue(value, slotIndex * entryCountPerOperation + index, config.seed);
            if (useDeviceBuffers) {
                const auto offset = AlignUp(deviceBufferSize, DeviceBufferAlignment());
                if (offset < deviceBufferSize ||
                    value.size() > std::numeric_limits<std::size_t>::max() - offset) {
                    return Status::Error(kExitInvalidArgument,
                                         "device payload bench buffer size overflow");
                }
                buffers.deviceBufferOffsets.emplace_back(offset);
                deviceBufferSize = offset + value.size();
            }
        }
        if (useDeviceBuffers) {
            const auto registerAlignment = DeviceAllocationAlignment(allocationPolicy);
            const auto registerSize = AlignUp(deviceBufferSize, registerAlignment);
            if (registerSize < deviceBufferSize) {
                return Status::Error(kExitInvalidArgument,
                                     "device payload bench register size overflow");
            }
            std::shared_ptr<void> deviceBuffer;
            auto status = AllocateDeviceBuffer(registerSize, allocationPolicy, deviceBuffer);
            if (!status.Ok()) { return status; }
            buffers.deviceBuffers.emplace_back(std::move(deviceBuffer));
        }
    }
    return Status::Success();
}

Status SyncStoreBenchDeviceBuffers(const KvTestConfig& config, BenchBufferPool& pool,
                                   std::size_t entryCount)
{
    for (auto& slot : pool) {
        auto status = SyncBenchDeviceBuffers(config, slot, entryCount);
        if (!status.Ok()) { return status; }
    }
    return Status::Success();
}

bool IsBenchReadOperation(BenchOpType op, std::uint64_t operationIndex, const BenchConfig& bench)
{
    if (op == BenchOpType::RETRIEVE || op == BenchOpType::BATCH_RETRIEVE) { return true; }
    if (op == BenchOpType::STORE || op == BenchOpType::BATCH_STORE) { return false; }

    const auto ratioTotal = bench.readRatio + bench.writeRatio;
    if (ratioTotal == 0) { return true; }
    return operationIndex % ratioTotal < bench.readRatio;
}

Status PrepareBenchBuffers(BenchBufferSlot& slot, std::uint64_t begin, std::size_t entryCount,
                           const std::string& keyPrefix, bool useDeviceBuffers,
                           DeviceAllocationPolicy allocationPolicy, std::int32_t logicalDeviceId)
{
    auto& buffers = slot.buffers;
    buffers.regions.clear();
    buffers.entries.clear();
    buffers.entryRegionIndexes.clear();
    buffers.entries.reserve(entryCount);
    if (useDeviceBuffers) {
        const auto baseAddr =
            buffers.deviceBuffers.empty()
                ? 0
                : reinterpret_cast<std::uintptr_t>(buffers.deviceBuffers.front().get());
        const auto dataSize = buffers.ownedBuffers.empty() ? 0
                                                           : buffers.deviceBufferOffsets.back() +
                                                                 buffers.ownedBuffers.back().size();
        const auto regionAlignment = DeviceAllocationAlignment(allocationPolicy);
        const auto regionSize = AlignUp(dataSize, regionAlignment);
        buffers.regions.emplace_back(MakeDeviceRegion(baseAddr, regionSize, logicalDeviceId));
        buffers.entryRegionIndexes.assign(entryCount, 0);
    } else {
        buffers.regions.reserve(entryCount);
    }

    for (std::size_t index = 0; index < entryCount; ++index) {
        const auto keyIndex = begin + index;
        const auto region =
            useDeviceBuffers
                ? MakeDeviceRegion(
                      reinterpret_cast<std::uintptr_t>(buffers.deviceBuffers.front().get()) +
                          buffers.deviceBufferOffsets[index],
                      buffers.ownedBuffers[index].size(), logicalDeviceId)
                : MakeHostRegion(buffers.ownedBuffers[index]);
        if (!useDeviceBuffers) { buffers.regions.emplace_back(region); }
        kv::CacheKey key{};
        auto status =
            StringToCacheKey(keyPrefix + std::to_string(keyIndex), "bench generated", key);
        if (!status.Ok()) { return status; }
        buffers.entries.emplace_back(MakeKvBuffer(key, region));
    }
    return Status::Success();
}

Status BindRegisteredBuffers(BufferSet& buffers)
{
    if (buffers.registeredRegions.size() != buffers.regions.size()) {
        return Status::Error(kExitInvalidArgument,
                             "registered buffer result count does not match region count");
    }
    for (std::size_t entryIndex = 0; entryIndex < buffers.entries.size(); ++entryIndex) {
        const auto regionIndex = buffers.entryRegionIndexes.empty()
                                     ? entryIndex
                                     : buffers.entryRegionIndexes[entryIndex];
        if (regionIndex >= buffers.registeredRegions.size()) {
            return Status::Error(kExitInvalidArgument,
                                 "registered buffer region index out of range");
        }
        buffers.entries[entryIndex].buffer.handle = buffers.registeredRegions[regionIndex].handle;
    }
    return Status::Success();
}

Status RegisterBenchDeviceBuffers(AsuClientRunner& clientRunner, BenchBufferPool& pool,
                                  std::uint64_t entryCountPerOperation,
                                  const std::string& keyPrefix,
                                  DeviceAllocationPolicy allocationPolicy,
                                  std::int32_t logicalDeviceId)
{
    for (std::size_t slotIndex = 0; slotIndex < pool.size(); ++slotIndex) {
        auto& slot = pool[slotIndex];
        auto status =
            PrepareBenchBuffers(slot, slotIndex * entryCountPerOperation,
                                static_cast<std::size_t>(entryCountPerOperation), keyPrefix,
                                /*useDeviceBuffers=*/true, allocationPolicy, logicalDeviceId);
        if (!status.Ok()) { return status; }
        status = clientRunner.RegisterBuffers(slot.buffers);
        if (!status.Ok()) { return status; }
    }
    return Status::Success();
}

Status UnregisterBenchDeviceBuffers(AsuClientRunner& clientRunner, BenchBufferPool& pool)
{
    Status finalStatus = Status::Success();
    for (auto& slot : pool) {
        auto& buffers = slot.buffers;
        if (buffers.registeredRegions.empty()) { continue; }
        auto status = clientRunner.UnregisterBuffers(buffers);
        if (finalStatus.Ok() && !status.Ok()) { finalStatus = status; }
        buffers.registeredRegions.clear();
    }
    return finalStatus;
}

Status ExecuteBenchOperation(BenchOpType requestedOp, const KvTestConfig& config,
                             AsuClientRunner& clientRunner, BenchBufferSlot& slot,
                             std::uint64_t begin, std::size_t entryCount,
                             std::uint64_t operationIndex, const std::string& keyPrefix,
                             bool useDeviceBuffers, CommandResult& operationResult)
{
    const auto& bench = config.bench;
    const bool isRead = IsBenchReadOperation(requestedOp, operationIndex, bench);
    const auto allocationPolicy = AllocationPolicyForConfig(config);
    const auto logicalDeviceId = ResolvePayloadDeviceId(config);
    const auto submitMode =
        entryCount > 1 ? SubmitMode::ALL_ENTRIES_IN_ONE_CALL : SubmitMode::SINGLE_ENTRY_PER_CALL;
    auto status = PrepareBenchBuffers(slot, begin, entryCount, keyPrefix, useDeviceBuffers,
                                      allocationPolicy, logicalDeviceId);
    if (!status.Ok()) { return status; }
    auto& buffers = slot.buffers;

    if (useDeviceBuffers && !isRead && requestedOp == BenchOpType::MIX) {
        auto status = SyncBenchDeviceBuffers(config, slot, entryCount);
        if (!status.Ok()) { return status; }
    }

    status =
        useDeviceBuffers ? BindRegisteredBuffers(buffers) : clientRunner.RegisterBuffers(buffers);
    if (!status.Ok()) { return status; }

    status =
        isRead ? clientRunner.Retrieve(buffers, submitMode,
                                       config.asuClientConfig.defaultWaitTimeoutMs, operationResult)
               : clientRunner.Store(buffers, submitMode,
                                    config.asuClientConfig.defaultWaitTimeoutMs, operationResult);
    if (useDeviceBuffers) { return status; }

    auto unregisterStatus = clientRunner.UnregisterBuffers(buffers);
    if (status.Ok() && !unregisterStatus.Ok()) { status = unregisterStatus; }
    return status;
}

OperationOutcome RunBenchOperation(BenchOpType op, const KvTestConfig& config,
                                   AsuClientRunner& clientRunner, BenchBufferSlot& slot,
                                   std::uint64_t begin, std::size_t entryCount,
                                   std::uint64_t operationIndex, const std::string& keyPrefix,
                                   bool useDeviceBuffers)
{
    CommandResult operationResult;
    const auto operationStart = std::chrono::steady_clock::now();
    auto opStatus =
        ExecuteBenchOperation(op, config, clientRunner, slot, begin, entryCount, operationIndex,
                              keyPrefix, useDeviceBuffers, operationResult);
    const auto operationEnd = std::chrono::steady_clock::now();

    OperationOutcome outcome;
    outcome.status = opStatus;
    outcome.latencyUs =
        std::chrono::duration<double, std::micro>(operationEnd - operationStart).count();
    outcome.entryCount = entryCount;
    outcome.bytes = entryCount * config.bench.ioSize;
    return outcome;
}

}  // namespace

Status BenchRunner::Run(const CommandOptions& options, const KvTestConfig& config,
                        AsuClientRunner& clientRunner, CommandResult& result) const
{
    const auto& bench = config.bench;
    if (bench.op == BenchOpType::UNKNOWN) {
        return Status::Error(kExitInvalidArgument, "bench op is required");
    }
    if (bench.concurrency == 0) {
        return Status::Error(kExitInvalidArgument, "bench concurrency must be greater than zero");
    }
    if (bench.durationSec == 0 && bench.ioCount == 0) {
        return Status::Error(kExitInvalidArgument,
                             "bench duration or IO count must be greater than zero");
    }
    if (bench.op == BenchOpType::MIX && bench.readRatio + bench.writeRatio == 0) {
        return Status::Error(kExitInvalidArgument,
                             "bench mix requires read_ratio or write_ratio greater than zero");
    }
    if (bench.readRatio > 100 || bench.writeRatio > 100) {
        return Status::Error(kExitInvalidArgument,
                             "bench read_ratio and write_ratio must be in range 0..100");
    }
    if (bench.op == BenchOpType::MIX && bench.readRatio + bench.writeRatio != 100) {
        return Status::Error(kExitInvalidArgument,
                             "bench mix read_ratio and write_ratio must sum to 100");
    }

    const auto entryCountPerOperation = BenchEntryCount(bench, bench.op);
    const auto poolEntryCount = entryCountPerOperation * bench.concurrency;
    const auto keyCount = std::max<std::uint64_t>(
        config.count, std::max<std::uint64_t>(bench.concurrency * entryCountPerOperation * 16,
                                              entryCountPerOperation));

    auto status = ValidateBenchBufferConfig(bench, poolEntryCount, config.memoryMaxBytes);
    if (!status.Ok()) { return status; }
    if (bench.ioCount != 0 &&
        bench.ioCount > std::numeric_limits<std::uint64_t>::max() / bench.ioSize) {
        return Status::Error(kExitInvalidArgument, "bench total IO bytes overflow uint64");
    }

    const std::string keyPrefix = config.keyPrefix.empty() ? "b" : config.keyPrefix;
    const bool useDeviceBuffers = UsesDevicePayloadBuffers(config);
    const auto allocationPolicy = AllocationPolicyForConfig(config);
    const auto logicalDeviceId = ResolvePayloadDeviceId(config);
    BenchBufferPool bufferPool;
    status = BuildBenchBufferPool(config, useDeviceBuffers, entryCountPerOperation, bufferPool);
    if (!status.Ok()) { return status; }
    if (useDeviceBuffers &&
        (bench.op == BenchOpType::STORE || bench.op == BenchOpType::BATCH_STORE)) {
        status = SyncStoreBenchDeviceBuffers(config, bufferPool,
                                             static_cast<std::size_t>(entryCountPerOperation));
        if (!status.Ok()) { return status; }
    }
    if (useDeviceBuffers) {
        status = RegisterBenchDeviceBuffers(clientRunner, bufferPool, entryCountPerOperation,
                                            keyPrefix, allocationPolicy, logicalDeviceId);
        if (!status.Ok()) {
            (void)UnregisterBenchDeviceBuffers(clientRunner, bufferPool);
            return status;
        }
    }

    std::unique_ptr<BenchExecutor> executor;
    if (bench.concurrency > 1) {
        try {
            executor = std::make_unique<BenchExecutor>(bench.concurrency);
        } catch (const std::system_error& error) {
            if (useDeviceBuffers) { (void)UnregisterBenchDeviceBuffers(clientRunner, bufferPool); }
            return Status::Error(kExitInvalidArgument,
                                 "bench worker creation failed: " + std::string(error.what()));
        }
    }

    using Clock = std::chrono::steady_clock;
    std::vector<double> measuredLatenciesUs;
    std::uint64_t operationIndex = 0;
    result = CommandResult{};
    result.benchMetrics.op = bench.op;
    result.benchMetrics.valueSize = bench.ioSize;
    result.benchMetrics.batchSize = static_cast<std::uint32_t>(entryCountPerOperation);
    result.benchMetrics.concurrency = bench.concurrency;
    result.benchMetrics.warmupSec = bench.warmupSec;
    result.benchMetrics.durationSec = bench.durationSec;

    auto runPhase = [&](std::uint64_t durationSec, std::uint64_t ioCount,
                        bool collectStats) -> Status {
        const auto phaseStart = Clock::now();
        const auto phaseEnd = phaseStart + std::chrono::seconds(durationSec);
        std::uint64_t scheduledEntryCount = 0;
        std::uint64_t windowOperationCount = 0;
        std::uint64_t windowEntryCount = 0;
        std::uint64_t windowBytes = 0;
        std::uint64_t windowErrors = 0;
        double windowLatencyUs = 0.0;
        std::uint64_t currentSecond = 1;

        auto emitProgressSample = [&](std::uint64_t operationsPerSec) {
            BenchRealtimeSample sample;
            sample.timestampSec = currentSecond;
            sample.op = bench.op;
            sample.bandwidthBytesPerSec = static_cast<double>(windowBytes);
            sample.iops = static_cast<double>(windowEntryCount);
            sample.avgLatencyUs = windowOperationCount == 0
                                      ? 0.0
                                      : windowLatencyUs / static_cast<double>(windowOperationCount);
            sample.errorCount = windowErrors;
            result.benchMetrics.realtimeSamples.emplace_back(sample);
            if (options.progress) { PrintProgressSample(sample, operationsPerSec); }
        };

        auto shouldContinue = [&]() {
            return ioCount == 0 ? Clock::now() < phaseEnd : scheduledEntryCount < ioCount;
        };

        while (shouldContinue()) {
            std::vector<std::future<OperationOutcome>> futures;
            std::vector<OperationOutcome> inlineOutcomes;
            futures.reserve(bench.concurrency);
            inlineOutcomes.reserve(bench.concurrency);
            for (std::uint32_t inFlight = 0; inFlight < bench.concurrency && shouldContinue();
                 ++inFlight) {
                const auto begin =
                    static_cast<std::size_t>((operationIndex * entryCountPerOperation) % keyCount);
                const auto available = keyCount - begin;
                const auto remainingEntryCount =
                    ioCount == 0 ? entryCountPerOperation : ioCount - scheduledEntryCount;
                const auto currentEntryCount = static_cast<std::size_t>(
                    std::min({entryCountPerOperation, static_cast<std::uint64_t>(available),
                              remainingEntryCount}));
                scheduledEntryCount += currentEntryCount;
                const auto currentOperationIndex = operationIndex++;
                auto* bufferSlot = &bufferPool[inFlight];

                if (bench.concurrency == 1) {
                    inlineOutcomes.emplace_back(RunBenchOperation(
                        bench.op, config, clientRunner, *bufferSlot, begin, currentEntryCount,
                        currentOperationIndex, keyPrefix, useDeviceBuffers));
                    continue;
                }

                try {
                    futures.emplace_back(executor->Submit([&, begin, currentEntryCount,
                                                           currentOperationIndex,
                                                           bufferSlot]() -> OperationOutcome {
                        return RunBenchOperation(bench.op, config, clientRunner, *bufferSlot, begin,
                                                 currentEntryCount, currentOperationIndex,
                                                 keyPrefix, useDeviceBuffers);
                    }));
                } catch (const std::exception& error) {
                    return Status::Error(
                        kExitInvalidArgument,
                        "bench operation submission failed: " + std::string(error.what()));
                }
            }

            for (auto& outcome : inlineOutcomes) {
                if (!outcome.status.Ok()) {
                    ++result.benchMetrics.errorCount;
                    if (collectStats) { ++windowErrors; }
                    result.status = outcome.status;
                    return outcome.status;
                }

                if (!collectStats) { continue; }

                measuredLatenciesUs.push_back(outcome.latencyUs);
                ++result.benchMetrics.completedOperations;
                result.benchMetrics.completedEntries += outcome.entryCount;
                result.benchMetrics.completedBytes += outcome.bytes;
                ++windowOperationCount;
                windowEntryCount += outcome.entryCount;
                windowBytes += outcome.bytes;
                windowLatencyUs += outcome.latencyUs;

                const auto operationEnd = Clock::now();
                const auto elapsedSec =
                    std::chrono::duration_cast<std::chrono::seconds>(operationEnd - phaseStart)
                        .count() +
                    1;
                if (static_cast<std::uint64_t>(elapsedSec) != currentSecond) {
                    emitProgressSample(windowOperationCount);

                    currentSecond = static_cast<std::uint64_t>(elapsedSec);
                    windowOperationCount = 0;
                    windowEntryCount = 0;
                    windowBytes = 0;
                    windowErrors = 0;
                    windowLatencyUs = 0.0;
                }
            }

            for (auto& future : futures) {
                auto outcome = future.get();
                if (!outcome.status.Ok()) {
                    ++result.benchMetrics.errorCount;
                    if (collectStats) { ++windowErrors; }
                    result.status = outcome.status;
                    return outcome.status;
                }

                if (!collectStats) { continue; }

                measuredLatenciesUs.push_back(outcome.latencyUs);
                ++result.benchMetrics.completedOperations;
                result.benchMetrics.completedEntries += outcome.entryCount;
                result.benchMetrics.completedBytes += outcome.bytes;
                ++windowOperationCount;
                windowEntryCount += outcome.entryCount;
                windowBytes += outcome.bytes;
                windowLatencyUs += outcome.latencyUs;

                const auto operationEnd = Clock::now();
                const auto elapsedSec =
                    std::chrono::duration_cast<std::chrono::seconds>(operationEnd - phaseStart)
                        .count() +
                    1;
                if (static_cast<std::uint64_t>(elapsedSec) != currentSecond) {
                    emitProgressSample(windowOperationCount);

                    currentSecond = static_cast<std::uint64_t>(elapsedSec);
                    windowOperationCount = 0;
                    windowEntryCount = 0;
                    windowBytes = 0;
                    windowErrors = 0;
                    windowLatencyUs = 0.0;
                }
            }
        }

        if (collectStats && (windowOperationCount != 0 || windowErrors != 0)) {
            emitProgressSample(windowOperationCount);
        }

        return Status::Success();
    };

    if (bench.warmupSec != 0) {
        status = runPhase(bench.warmupSec, 0, false);
        if (!status.Ok()) {
            executor.reset();
            if (useDeviceBuffers) { (void)UnregisterBenchDeviceBuffers(clientRunner, bufferPool); }
            return status;
        }
    }

    const auto measureStart = Clock::now();
    status = runPhase(bench.durationSec, bench.ioCount, true);
    const auto measureEnd = Clock::now();
    executor.reset();
    auto cleanupStatus = useDeviceBuffers ? UnregisterBenchDeviceBuffers(clientRunner, bufferPool)
                                          : Status::Success();
    if (status.Ok() && !cleanupStatus.Ok()) { status = cleanupStatus; }
    if (!status.Ok()) { return status; }

    result.benchMetrics.elapsedSec =
        std::chrono::duration<double>(measureEnd - measureStart).count();
    if (result.benchMetrics.elapsedSec > 0.0) {
        result.benchMetrics.avgBandwidthBytesPerSec =
            static_cast<double>(result.benchMetrics.completedBytes) /
            result.benchMetrics.elapsedSec;
        result.benchMetrics.avgIops = static_cast<double>(result.benchMetrics.completedEntries) /
                                      result.benchMetrics.elapsedSec;
        result.benchMetrics.avgBatchIops =
            static_cast<double>(result.benchMetrics.completedOperations) /
            result.benchMetrics.elapsedSec;
    }
    result.benchMetrics.latency = BuildLatencyStats(measuredLatenciesUs);
    result.taskResult = BuildEmptyTaskResult();
    result.status = Status::Success();
    return result.status;
}

}  // namespace kv::bench
