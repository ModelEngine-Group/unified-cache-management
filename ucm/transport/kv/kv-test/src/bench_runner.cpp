#include "kv_test/bench_runner.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <numeric>
#include "kv_test/buffer_allocator.h"

namespace UC::KVTest {

namespace {

constexpr int kExitInvalidArgument = 1;

UC::ASU::MemoryRegion MakeHostRegion(std::vector<std::uint8_t>& buffer)
{
    UC::ASU::MemoryRegion region;
    region.memoryType = UC::ASU::MemoryType::HOST;
    region.addr = buffer.empty() ? 0 : reinterpret_cast<std::uint64_t>(buffer.data());
    region.size = buffer.size();
    region.deviceId = -1;
    region.numaNode = -1;
    return region;
}

UC::ASU::KVBuffer MakeKvBuffer(const UC::ASU::CacheKey& key, const UC::ASU::MemoryRegion& region)
{
    UC::ASU::Buffer buffer;
    buffer.region = region;
    buffer.handle = UC::ASU::kInvalidMRHandle;
    return UC::ASU::KVBuffer{key, buffer};
}

UC::ASU::TaskResult BuildEmptyTaskResult()
{
    UC::ASU::TaskResult result;
    result.status = UC::ASU::Status::OK();
    return result;
}

struct OperationOutcome {
    Status status;
    double latencyUs{0.0};
    std::size_t entryCount{0};
    std::uint64_t bytes{0};
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

Status CheckBenchMemoryLimit(std::uint64_t keyCount, std::uint64_t ioSize,
                             std::uint64_t memoryMaxBytes)
{
    if (memoryMaxBytes == 0) {
        return Status::Error(kExitInvalidArgument,
                             "limits.memory_max_bytes must be greater than zero");
    }
    if (keyCount == 0 || ioSize == 0) { return Status::Success(); }
    if (keyCount > std::numeric_limits<std::uint64_t>::max() / ioSize) {
        return Status::Error(kExitInvalidArgument, "bench generated value bytes overflow uint64");
    }

    const auto requiredBytes = keyCount * ioSize;
    if (requiredBytes > memoryMaxBytes) {
        return Status::Error(
            kExitInvalidArgument,
            "bench generated value bytes exceed limits.memory_max_bytes: required=" +
                std::to_string(requiredBytes) + ", limit=" + std::to_string(memoryMaxBytes));
    }
    return Status::Success();
}

Status BuildBenchData(const KvTestConfig& config, std::uint64_t keyCount, GeneratedData& data)
{
    if (config.bench.ioSize == 0) {
        return Status::Error(kExitInvalidArgument, "bench io_size must be greater than zero");
    }
    if (keyCount > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Status::Error(kExitInvalidArgument, "bench key count exceeds addressable memory");
    }
    if (config.bench.ioSize > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Status::Error(kExitInvalidArgument, "bench io_size exceeds addressable memory");
    }

    auto status = CheckBenchMemoryLimit(keyCount, config.bench.ioSize, config.memoryMaxBytes);
    if (!status.Ok()) { return status; }

    const std::string keyPrefix = config.keyPrefix.empty() ? "bench-key-" : config.keyPrefix;
    data = GeneratedData{};
    data.keys.reserve(static_cast<std::size_t>(keyCount));
    data.values.reserve(static_cast<std::size_t>(keyCount));
    for (std::uint64_t index = 0; index < keyCount; ++index) {
        data.keys.emplace_back(keyPrefix + std::to_string(index));
        auto& value = data.values.emplace_back(config.bench.ioSize);
        for (std::uint64_t byteIndex = 0; byteIndex < config.bench.ioSize; ++byteIndex) {
            value[byteIndex] = static_cast<std::uint8_t>((index + byteIndex + config.seed) & 0xFF);
        }
    }
    return Status::Success();
}

BufferSet SliceBuffers(const BufferSet& source, std::size_t begin, std::size_t count)
{
    BufferSet slice;
    const auto end = std::min(begin + count, source.entries.size());
    slice.ownedBuffers.reserve(end - begin);
    slice.regions.reserve(end - begin);
    slice.entries.reserve(end - begin);

    for (auto index = begin; index < end; ++index) {
        slice.ownedBuffers.emplace_back(source.ownedBuffers[index]);
    }
    for (std::size_t index = 0; index < slice.ownedBuffers.size(); ++index) {
        auto region = MakeHostRegion(slice.ownedBuffers[index]);
        slice.regions.emplace_back(region);
        slice.entries.emplace_back(MakeKvBuffer(source.entries[begin + index].key, region));
    }
    return slice;
}

bool IsBenchReadOperation(BenchOpType op, std::uint64_t operationIndex, const BenchConfig& bench)
{
    if (op == BenchOpType::RETRIEVE || op == BenchOpType::BATCH_RETRIEVE) { return true; }
    if (op == BenchOpType::STORE || op == BenchOpType::BATCH_STORE) { return false; }

    const auto ratioTotal = bench.readRatio + bench.writeRatio;
    if (ratioTotal == 0) { return true; }
    return operationIndex % ratioTotal < bench.readRatio;
}

Status ExecuteBenchOperation(BenchOpType requestedOp, const BenchConfig& bench,
                             AsuClientRunner& clientRunner, const BufferSet& storeBuffers,
                             const BufferSet& retrieveBuffers, std::size_t begin,
                             std::size_t entryCount, std::uint64_t operationIndex,
                             std::uint64_t timeoutMs, CommandResult& operationResult)
{
    const bool isRead = IsBenchReadOperation(requestedOp, operationIndex, bench);
    const auto submitMode =
        entryCount > 1 ? SubmitMode::ALL_ENTRIES_IN_ONE_CALL : SubmitMode::SINGLE_ENTRY_PER_CALL;
    auto buffers = SliceBuffers(isRead ? retrieveBuffers : storeBuffers, begin, entryCount);

    auto status = clientRunner.RegisterBuffers(buffers);
    if (!status.Ok()) { return status; }

    status = isRead ? clientRunner.Retrieve(buffers, submitMode, timeoutMs, operationResult)
                    : clientRunner.Store(buffers, submitMode, timeoutMs, operationResult);
    auto unregisterStatus = clientRunner.UnregisterBuffers(buffers);
    if (status.Ok() && !unregisterStatus.Ok()) { status = unregisterStatus; }
    return status;
}

using EntrySubmitMethod = UC::ASU::Status (UC::ASU::AsuClient::*)(
    const std::vector<UC::ASU::KVBuffer>&, UC::ASU::TaskId&);

}  // namespace

Status BenchRunner::Run(const CommandOptions&, const KvTestConfig& config,
                        AsuClientRunner& clientRunner, CommandResult& result) const
{
    const auto& bench = config.bench;
    if (bench.op == BenchOpType::UNKNOWN) {
        return Status::Error(kExitInvalidArgument, "bench op is required");
    }
    if (bench.concurrency == 0) {
        return Status::Error(kExitInvalidArgument, "bench concurrency must be greater than zero");
    }
    if (bench.durationSec == 0) {
        return Status::Error(kExitInvalidArgument, "bench duration must be greater than zero");
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
    const auto keyCount = std::max<std::uint64_t>(
        config.count, std::max<std::uint64_t>(bench.concurrency * entryCountPerOperation * 16,
                                              entryCountPerOperation));

    GeneratedData data;
    auto status = BuildBenchData(config, keyCount, data);
    if (!status.Ok()) { return status; }

    BufferAllocator allocator;
    BufferSet storeBuffers;
    status = allocator.BuildStoreBuffers(data, storeBuffers);
    if (!status.Ok()) { return status; }

    BufferSet retrieveBuffers;
    status = allocator.BuildRetrieveBuffers(data, retrieveBuffers);
    if (!status.Ok()) { return status; }

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

    auto runPhase = [&](std::uint64_t durationSec, bool collectStats) -> Status {
        const auto phaseStart = Clock::now();
        const auto phaseEnd = phaseStart + std::chrono::seconds(durationSec);
        std::uint64_t windowOperationCount = 0;
        std::uint64_t windowEntryCount = 0;
        std::uint64_t windowBytes = 0;
        std::uint64_t windowErrors = 0;
        double windowLatencyUs = 0.0;
        std::uint64_t currentSecond = 1;

        while (Clock::now() < phaseEnd) {
            std::vector<std::future<OperationOutcome>> futures;
            futures.reserve(bench.concurrency);
            for (std::uint32_t inFlight = 0;
                 inFlight < bench.concurrency && Clock::now() < phaseEnd; ++inFlight) {
                const auto begin =
                    static_cast<std::size_t>((operationIndex * entryCountPerOperation) % keyCount);
                const auto available = keyCount - begin;
                const auto currentEntryCount = static_cast<std::size_t>(
                    std::min<std::uint64_t>(entryCountPerOperation, available));
                const auto currentOperationIndex = operationIndex++;

                futures.emplace_back(std::async(
                    std::launch::async,
                    [&, begin, currentEntryCount, currentOperationIndex]() -> OperationOutcome {
                        CommandResult operationResult;
                        const auto operationStart = Clock::now();
                        auto opStatus = ExecuteBenchOperation(
                            bench.op, bench, clientRunner, storeBuffers, retrieveBuffers, begin,
                            currentEntryCount, currentOperationIndex,
                            config.asuClientConfig.defaultWaitTimeoutMs, operationResult);
                        const auto operationEnd = Clock::now();
                        OperationOutcome outcome;
                        outcome.status = opStatus;
                        outcome.latencyUs =
                            std::chrono::duration<double, std::micro>(operationEnd - operationStart)
                                .count();
                        outcome.entryCount = currentEntryCount;
                        outcome.bytes = currentEntryCount * bench.ioSize;
                        return outcome;
                    }));
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
                    BenchRealtimeSample sample;
                    sample.timestampSec = currentSecond;
                    sample.op = bench.op;
                    sample.bandwidthBytesPerSec = static_cast<double>(windowBytes);
                    sample.iops = static_cast<double>(windowEntryCount);
                    sample.avgLatencyUs =
                        windowOperationCount == 0
                            ? 0.0
                            : windowLatencyUs / static_cast<double>(windowOperationCount);
                    sample.errorCount = windowErrors;
                    result.benchMetrics.realtimeSamples.emplace_back(sample);

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
        }

        return Status::Success();
    };

    if (bench.warmupSec != 0) {
        status = runPhase(bench.warmupSec, false);
        if (!status.Ok()) { return status; }
    }

    const auto measureStart = Clock::now();
    status = runPhase(bench.durationSec, true);
    const auto measureEnd = Clock::now();
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

}  // namespace UC::KVTest
