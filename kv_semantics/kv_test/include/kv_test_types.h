#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "kv_client.h"

namespace UC::KVTest {

constexpr std::uint64_t kDefaultMemoryMaxBytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;

enum class CommandType {
    CONNECT = 0,
    CONFIG_CHECK,
    STORE,
    RETRIEVE,
    DELETE,
    EXIST,
    BATCH_STORE,
    BATCH_RETRIEVE,
    POWER_CYCLE_PREPARE,
    POWER_CYCLE_VERIFY,
    BENCH,
    VERSION,
    UNKNOWN,
};

enum class BenchOpType {
    STORE = 0,
    RETRIEVE,
    BATCH_STORE,
    BATCH_RETRIEVE,
    MIX,
    UNKNOWN,
};

enum class ConfigFormat {
    ASU_CLIENT_KEY_VALUE = 0,
};

enum class HcommApiBoundary {
    C_API = 0,
};

enum class WireProtocol {
    SQE = 0,
};

enum class ValuePlacement {
    ASU_MANAGED = 0,
};

enum class PayloadBufferPlacement {
    HOST = 0,
    ASCEND_DEVICE,
};

enum class DeviceAllocationPolicy {
    DEFAULT = 0,
    AIV_REGISTERABLE,
};

enum class DigestAlgorithm {
    CRC64 = 0,
};

enum class SubmitMode {
    SINGLE_ENTRY_PER_CALL = 0,
    ALL_ENTRIES_IN_ONE_CALL,
};

enum class RetrieveMissingKeyPolicy {
    PARTIAL_FAILED = 0,
};

enum class TransportLinkPolicy {
    SHARED_DATA_LINK = 0,
};

enum class HcommLocalRole {
    SERVER = 0,
};

enum class HcommProtocol {
    UBOE = 0,
    ROCE,
    PCIE,
};

enum class HcommChannelConfigSource {
    ASU_CONFIG = 0,
};

struct Status {
    int code{0};
    std::string message;

    bool Ok() const noexcept { return code == 0; }

    static Status Success() { return {}; }
    static Status Error(int errorCode, std::string errorMessage)
    {
        return Status{errorCode, std::move(errorMessage)};
    }
};

struct CommandOptions {
    CommandType command{CommandType::UNKNOWN};
    BenchOpType benchOp{BenchOpType::UNKNOWN};
    std::string configPath;
    std::vector<std::string> keys;
    std::string keysFile;
    std::string keyPrefix;
    std::uint64_t count{0};
    std::uint64_t ioCount{0};
    std::uint64_t keyStart{0};
    std::uint64_t keyEnd{0};
    std::uint64_t seed{0};
    std::uint64_t valueSize{0};
    std::uint32_t batchSize{0};
    std::uint32_t concurrency{0};
    std::uint64_t durationSec{0};
    std::uint64_t warmupSec{0};
    std::uint32_t readRatio{0};
    std::uint32_t writeRatio{0};
    std::uint64_t timeoutMs{0};
    bool check{false};
    bool helpRequested{false};
    bool versionRequested{false};
    bool singleKeyRequested{false};
    bool keyStartSet{false};
    bool keyEndSet{false};
    bool progress{false};
    std::string outputPath;
};

struct BenchConfig {
    BenchOpType op{BenchOpType::UNKNOWN};
    std::uint64_t ioSize{0};
    std::uint32_t concurrency{0};
    std::uint64_t durationSec{0};
    std::uint64_t ioCount{0};
    std::uint64_t warmupSec{0};
    std::uint32_t readRatio{0};
    std::uint32_t writeRatio{0};
    std::uint32_t batchSize{0};
};

struct OutputConfig {
    std::string path;
    std::uint64_t realtimeFileMaxBytes{0};
};

struct KvTestFakeBackendConfig {
    std::string storePath;
    std::uint64_t latencyMs{1};
    std::uint64_t workerThreads{4};
};

struct AsuRuntimeLibraryConfig {
    std::string clientLibraryPath;
    std::string transportLibraryPath;
};

struct ToolBehaviorConfig {
    ConfigFormat configFormat{ConfigFormat::ASU_CLIENT_KEY_VALUE};
    HcommApiBoundary hcommApiBoundary{HcommApiBoundary::C_API};
    WireProtocol wireProtocol{WireProtocol::SQE};
    ValuePlacement valuePlacement{ValuePlacement::ASU_MANAGED};
    DigestAlgorithm digestAlgorithm{DigestAlgorithm::CRC64};
    RetrieveMissingKeyPolicy retrieveMissingKeyPolicy{RetrieveMissingKeyPolicy::PARTIAL_FAILED};
    TransportLinkPolicy transportLinkPolicy{TransportLinkPolicy::SHARED_DATA_LINK};
    HcommLocalRole hcommLocalRole{HcommLocalRole::SERVER};
    HcommChannelConfigSource hcommChannelConfigSource{HcommChannelConfigSource::ASU_CONFIG};
};

struct HcommProtocolMapping {
    // TODO: Verify UBOE and TCP->PCIE names against the final Hcomm public enum.
    HcommProtocol ub{HcommProtocol::UBOE};
    HcommProtocol roce{HcommProtocol::ROCE};
    HcommProtocol tcp{HcommProtocol::PCIE};
};

struct KvTestConfig {
    UC::ASU::AsuClientConfig asuClientConfig;
    ToolBehaviorConfig behavior;
    HcommProtocolMapping hcommProtocolMapping;
    BenchConfig bench;
    OutputConfig output;
    KvTestFakeBackendConfig fakeBackend;
    AsuRuntimeLibraryConfig asuRuntime;
    std::string keyPrefix;
    std::uint64_t seed{0};
    std::uint64_t valueSize{0};
    std::uint64_t count{0};
    std::uint64_t memoryMaxBytes{kDefaultMemoryMaxBytes};
};

struct GeneratedData {
    std::vector<UC::ASU::CacheKey> keys;
    std::vector<std::vector<std::uint8_t>> values;
};

struct BufferSet {
    std::vector<std::vector<std::uint8_t>> ownedBuffers;
    std::vector<std::shared_ptr<void>> deviceBuffers;
    std::vector<std::size_t> deviceBufferOffsets;
    std::vector<UC::ASU::MemoryRegion> regions;
    std::vector<UC::ASU::KVBuffer> entries;
    std::vector<std::size_t> entryRegionIndexes;
    std::vector<UC::ASU::RegisteredMemory> registeredRegions;
};

struct BenchLatencyStats {
    double avgUs{0.0};
    double minUs{0.0};
    double maxUs{0.0};
    double p99_9Us{0.0};
    double p99_99Us{0.0};
    double p99_999Us{0.0};
};

struct BenchRealtimeSample {
    std::uint64_t timestampSec{0};
    BenchOpType op{BenchOpType::UNKNOWN};
    double bandwidthBytesPerSec{0.0};
    double iops{0.0};
    double avgLatencyUs{0.0};
    std::uint64_t errorCount{0};
};

struct BenchMetrics {
    BenchOpType op{BenchOpType::UNKNOWN};
    std::uint64_t valueSize{0};
    std::uint32_t batchSize{0};
    std::uint32_t concurrency{0};
    std::uint64_t warmupSec{0};
    std::uint64_t durationSec{0};
    std::uint64_t completedOperations{0};
    std::uint64_t completedEntries{0};
    std::uint64_t completedBytes{0};
    std::uint64_t errorCount{0};
    double elapsedSec{0.0};
    double avgBandwidthBytesPerSec{0.0};
    double avgIops{0.0};
    double avgBatchIops{0.0};
    BenchLatencyStats latency;
    std::vector<BenchRealtimeSample> realtimeSamples;
};

struct ConsistencySummary {
    bool enabled{false};
    std::uint64_t checked{0};
    std::uint64_t passed{0};
    std::uint64_t failed{0};
    std::string key;
    std::string expected;
    std::string actual;
};

struct CommandResult {
    Status status;
    UC::ASU::TaskResult taskResult;
    UC::ASU::QueryResult queryResult;
    BenchMetrics benchMetrics;
    ConsistencySummary consistency;
};

}  // namespace UC::KVTest
