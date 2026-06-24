#include "kv_test/bench_runner.h"
#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include "kv_test/fake_backend.h"

namespace UC::KVTest {

namespace {

constexpr int kExitInvalidArgument = 1;

Status StringToCacheKey(const std::string& value, UC::ASU::CacheKey& key)
{
    if (value.size() > key.size()) {
        return Status::Error(kExitInvalidArgument,
                             "bench key length exceeds " + std::to_string(key.size()) +
                                 " bytes: length=" + std::to_string(value.size()) +
                                 ", key=" + value);
    }
    key = UC::ASU::CacheKey{};
    if (!value.empty()) { std::memcpy(key.data(), value.data(), value.size()); }
    return Status::Success();
}

struct BenchBufferSlot {
    BufferSet buffers;
};

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

UC::ASU::MemoryRegion MakeDeviceRegion(const std::shared_ptr<void>& buffer, std::size_t size)
{
    UC::ASU::MemoryRegion region;
    region.memoryType = UC::ASU::MemoryType::ASCEND_DEVICE;
    region.addr = buffer ? reinterpret_cast<std::uint64_t>(buffer.get()) : 0;
    region.size = size;
    region.deviceId = 0;
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

using BenchBufferPool = std::vector<BenchBufferSlot>;

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

std::string FormatMiBPerSec(double bytesPerSec)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << (bytesPerSec / (1024.0 * 1024.0));
    return stream.str();
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

Status CopyHostToDevice(const std::vector<std::uint8_t>& hostBuffer,
                        const std::shared_ptr<void>& deviceBuffer, std::size_t index)
{
    if (hostBuffer.empty()) { return Status::Success(); }
    auto ret = aclrtMemcpy(deviceBuffer.get(), hostBuffer.size(), hostBuffer.data(),
                           hostBuffer.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return Status::Error(
            kExitInvalidArgument,
            "fake_backend bench host-to-device copy failed: index=" + std::to_string(index) +
                " size=" + std::to_string(hostBuffer.size()) + " ret=" + std::to_string(ret));
    }
    return Status::Success();
}

Status MakeDeviceBuffer(const std::vector<std::uint8_t>& hostBuffer,
                        std::shared_ptr<void>& deviceBuffer)
{
    if (hostBuffer.empty()) {
        deviceBuffer.reset();
        return Status::Success();
    }

    void* ptr = nullptr;
    auto ret = aclrtMalloc(&ptr, hostBuffer.size(), ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    if (ret != ACL_SUCCESS) {
        return Status::Error(kExitInvalidArgument, "fake_backend bench aclrtMalloc failed: size=" +
                                                       std::to_string(hostBuffer.size()) +
                                                       " ret=" + std::to_string(ret));
    }
    deviceBuffer = std::shared_ptr<void>(ptr, aclrtFree);
    return CopyHostToDevice(hostBuffer, deviceBuffer, 0);
}

Status SyncBenchDeviceBuffers(const KvTestConfig& config, BenchBufferSlot& slot,
                              std::size_t entryCount)
{
    auto status = MaybeSetUpFakeBackendAclThread(config);
    if (!status.Ok()) { return status; }

    auto& buffers = slot.buffers;
    if (buffers.ownedBuffers.size() < entryCount || buffers.deviceBuffers.size() < entryCount) {
        return Status::Error(kExitInvalidArgument, "fake_backend bench buffer count mismatch");
    }
    for (std::size_t index = 0; index < entryCount; ++index) {
        status = CopyHostToDevice(buffers.ownedBuffers[index], buffers.deviceBuffers[index], index);
        if (!status.Ok()) { return status; }
    }
    return Status::Success();
}

Status BuildBenchBufferPool(const KvTestConfig& config, bool useDeviceBuffers,
                            std::uint64_t entryCountPerOperation, BenchBufferPool& pool)
{
    const auto& bench = config.bench;
    if (useDeviceBuffers) {
        auto status = MaybeSetUpFakeBackendAclThread(config);
        if (!status.Ok()) { return status; }
    }

    pool.resize(bench.concurrency);
    for (std::size_t slotIndex = 0; slotIndex < pool.size(); ++slotIndex) {
        auto& buffers = pool[slotIndex].buffers;
        buffers.ownedBuffers.reserve(static_cast<std::size_t>(entryCountPerOperation));
        buffers.deviceBuffers.reserve(static_cast<std::size_t>(entryCountPerOperation));
        for (std::uint64_t index = 0; index < entryCountPerOperation; ++index) {
            auto& value = buffers.ownedBuffers.emplace_back(static_cast<std::size_t>(bench.ioSize));
            FillStoreValue(value, slotIndex * entryCountPerOperation + index, config.seed);
            if (useDeviceBuffers) {
                std::shared_ptr<void> deviceBuffer;
                auto status = MakeDeviceBuffer(value, deviceBuffer);
                if (!status.Ok()) { return status; }
                buffers.deviceBuffers.emplace_back(std::move(deviceBuffer));
            }
        }
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
                           const std::string& keyPrefix, bool useDeviceBuffers)
{
    auto& buffers = slot.buffers;
    buffers.regions.clear();
    buffers.entries.clear();
    buffers.registerResults.clear();
    buffers.regions.reserve(entryCount);
    buffers.entries.reserve(entryCount);

    for (std::size_t index = 0; index < entryCount; ++index) {
        const auto keyIndex = begin + index;
        auto region = useDeviceBuffers ? MakeDeviceRegion(buffers.deviceBuffers[index],
                                                          buffers.ownedBuffers[index].size())
                                       : MakeHostRegion(buffers.ownedBuffers[index]);
        buffers.regions.emplace_back(region);
        UC::ASU::CacheKey key{};
        auto status = StringToCacheKey(keyPrefix + std::to_string(keyIndex), key);
        if (!status.Ok()) { return status; }
        buffers.entries.emplace_back(MakeKvBuffer(key, region));
    }
    return Status::Success();
}

Status ExecuteBenchOperation(BenchOpType requestedOp, const KvTestConfig& config,
                             AsuClientRunner& clientRunner, BenchBufferSlot& slot,
                             std::uint64_t begin, std::size_t entryCount,
                             std::uint64_t operationIndex, const std::string& keyPrefix,
                             bool useDeviceBuffers, CommandResult& operationResult)
{
    const auto& bench = config.bench;
    const bool isRead = IsBenchReadOperation(requestedOp, operationIndex, bench);
    const auto submitMode =
        entryCount > 1 ? SubmitMode::ALL_ENTRIES_IN_ONE_CALL : SubmitMode::SINGLE_ENTRY_PER_CALL;
    auto status = PrepareBenchBuffers(slot, begin, entryCount, keyPrefix, useDeviceBuffers);
    if (!status.Ok()) { return status; }
    auto& buffers = slot.buffers;

    if (useDeviceBuffers && !isRead) {
        auto status = SyncBenchDeviceBuffers(config, slot, entryCount);
        if (!status.Ok()) { return status; }
    }

    if (useDeviceBuffers && !isRead) {
        auto status = SyncBenchDeviceBuffers(config, slot, entryCount);
        if (!status.Ok()) { return status; }
    }

    status = clientRunner.RegisterBuffers(buffers);
    if (!status.Ok()) { return status; }

    status =
        isRead ? clientRunner.Retrieve(buffers, submitMode,
                                       config.asuClientConfig.defaultWaitTimeoutMs, operationResult)
               : clientRunner.Store(buffers, submitMode,
                                    config.asuClientConfig.defaultWaitTimeoutMs, operationResult);
    auto unregisterStatus = clientRunner.UnregisterBuffers(buffers);
    if (status.Ok() && !unregisterStatus.Ok()) { status = unregisterStatus; }
    return status;
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
    const auto poolEntryCount = entryCountPerOperation * bench.concurrency;
    const auto keyCount = std::max<std::uint64_t>(
        config.count, std::max<std::uint64_t>(bench.concurrency * entryCountPerOperation * 16,
                                              entryCountPerOperation));

    auto status = ValidateBenchBufferConfig(bench, poolEntryCount, config.memoryMaxBytes);
    if (!status.Ok()) { return status; }

    const std::string keyPrefix = config.keyPrefix.empty() ? "b" : config.keyPrefix;
    const bool useDeviceBuffers = IsFakeBackendMode(config);
    BenchBufferPool bufferPool;
    status = BuildBenchBufferPool(config, useDeviceBuffers, entryCountPerOperation, bufferPool);
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
                auto* bufferSlot = &bufferPool[inFlight];

                futures.emplace_back(std::async(
                    std::launch::async,
                    [&, begin, currentEntryCount, currentOperationIndex,
                     bufferSlot]() -> OperationOutcome {
                        CommandResult operationResult;
                        const auto operationStart = Clock::now();
                        auto opStatus = ExecuteBenchOperation(
                            bench.op, config, clientRunner, *bufferSlot, begin, currentEntryCount,
                            currentOperationIndex, keyPrefix, useDeviceBuffers, operationResult);
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
