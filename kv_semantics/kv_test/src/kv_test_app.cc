#include "kv_test_app.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include "asu_runtime_proxy.h"
#include "kv_test_config_helpers.h"
#include "payload_buffer_runtime.h"

namespace kv::bench {

namespace {

constexpr int kExitSuccess = 0;
constexpr int kExitInvalidArgument = 1;
constexpr const char* kAnsiGreen = "\033[32m";
constexpr const char* kAnsiRed = "\033[31m";
constexpr const char* kAnsiReset = "\033[0m";
int ToExitCode(const Status& status) { return status.Ok() ? kExitSuccess : status.code; }

std::filesystem::path PowerCycleMetadataPath(const KvTestConfig& config)
{
    const auto baseDir = config.output.path.empty() ? std::filesystem::path(".")
                                                    : std::filesystem::path(config.output.path);
    return baseDir / "power-cycle-metadata.conf";
}

std::uint64_t EffectiveSeed(const CommandOptions& options, const KvTestConfig& config)
{
    return options.seed == 0 ? config.seed : options.seed;
}

std::uint64_t EffectiveValueSize(const CommandOptions& options, const KvTestConfig& config)
{
    return options.valueSize == 0 ? config.valueSize : options.valueSize;
}

std::uint64_t EffectiveCount(const CommandOptions& options, const KvTestConfig& config)
{
    return options.count == 0 ? config.count : options.count;
}

std::string EffectiveKeyPrefix(const CommandOptions& options, const KvTestConfig& config)
{
    return options.keyPrefix.empty() ? config.keyPrefix : options.keyPrefix;
}

CommandOptions BuildEffectiveOptions(const CommandOptions& options, const KvTestConfig& config)
{
    CommandOptions effective = options;
    effective.seed = EffectiveSeed(options, config);
    effective.count = EffectiveCount(options, config);
    effective.valueSize = options.command == CommandType::BENCH
                              ? config.bench.ioSize
                              : EffectiveValueSize(options, config);
    effective.keyPrefix = EffectiveKeyPrefix(options, config);
    effective.batchSize = config.bench.batchSize;
    effective.timeoutMs = config.asuClientConfig.defaultWaitTimeoutMs;
    effective.outputPath = config.output.path;
    if (effective.benchOp == BenchOpType::UNKNOWN) { effective.benchOp = config.bench.op; }
    effective.concurrency = config.bench.concurrency;
    effective.durationSec = config.bench.durationSec;
    effective.ioCount = config.bench.ioCount;
    effective.warmupSec = config.bench.warmupSec;
    effective.readRatio = config.bench.readRatio;
    effective.writeRatio = config.bench.writeRatio;
    return effective;
}

Status WritePowerCycleMetadata(const CommandOptions& options, const KvTestConfig& config)
{
    std::error_code errorCode;
    auto metadataPath = PowerCycleMetadataPath(config);
    std::filesystem::create_directories(metadataPath.parent_path(), errorCode);
    if (errorCode) {
        return Status::Error(
            kExitInvalidArgument,
            "failed to create power-cycle metadata directory: " + errorCode.message());
    }

    std::ofstream file{metadataPath};
    if (!file.is_open()) {
        return Status::Error(kExitInvalidArgument,
                             "failed to open power-cycle metadata: " + metadataPath.string());
    }

    file << "key_prefix=" << EffectiveKeyPrefix(options, config) << '\n'
         << "seed=" << EffectiveSeed(options, config) << '\n'
         << "value_size=" << EffectiveValueSize(options, config) << '\n'
         << "count=" << EffectiveCount(options, config) << '\n';
    if (!file.good()) {
        return Status::Error(kExitInvalidArgument,
                             "failed to write power-cycle metadata: " + metadataPath.string());
    }
    return Status::Success();
}

bool ReadMetadataValue(const std::string& line, const std::string& key, std::string& value)
{
    const auto prefix = key + "=";
    if (line.rfind(prefix, 0) != 0) { return false; }
    value = line.substr(prefix.size());
    return true;
}

Status ValidatePowerCycleMetadata(const CommandOptions& options, const KvTestConfig& config)
{
    const auto metadataPath = PowerCycleMetadataPath(config);
    std::ifstream file{metadataPath};
    if (!file.is_open()) {
        return Status::Error(kExitInvalidArgument,
                             "failed to open power-cycle metadata: " + metadataPath.string());
    }

    std::unordered_map<std::string, std::string> values;
    std::string line;
    while (std::getline(file, line)) {
        for (const auto& key : {"key_prefix", "seed", "value_size", "count"}) {
            std::string value;
            if (ReadMetadataValue(line, key, value)) { values[key] = value; }
        }
    }

    const std::unordered_map<std::string, std::string> expected = {
        {"key_prefix", EffectiveKeyPrefix(options,                config) },
        {"seed",       std::to_string(EffectiveSeed(options,      config))},
        {"value_size", std::to_string(EffectiveValueSize(options, config))},
        {"count",      std::to_string(EffectiveCount(options,     config))},
    };
    for (const auto& item : expected) {
        auto iter = values.find(item.first);
        if (iter == values.end() || iter->second != item.second) {
            return Status::Error(kExitInvalidArgument,
                                 "power-cycle metadata mismatch for " + item.first);
        }
    }
    return Status::Success();
}

void PrintGeneralHelp()
{
    std::cout
        << "Usage:\n"
        << "  kv-test <command> [options]\n"
        << "  kv-test <command> --help\n"
        << "  kv-test --help\n\n"
        << "Config:\n"
        << "  --configpath <path>      Override config file path for this run.\n"
        << "  KV_TEST_CONFIG=<path>    Default config file path when --configpath is omitted.\n\n"
        << "Commands:\n"
        << "  connect\n"
        << "  config check\n"
        << "  version\n"
        << "  store | retrieve | delete | exist\n"
        << "  batch-store | batch-retrieve\n"
        << "  power-cycle prepare | power-cycle verify\n"
        << "  bench [store|retrieve|batch-store|batch-retrieve|mix]\n\n"
        << "Common options:\n"
        << "  --key <key>              Use one key.\n"
        << "  --keys <k1,k2,...>       Use a comma-separated key list.\n"
        << "  --keys-file <path>       Read comma-separated keys from a file.\n"
        << "  --count <n>              Generate n keys; for bench, run exactly n logical IOs.\n"
        << "  --prefix <p> --key-start <n> --key-end <n>\n"
        << "                           Generate keys in the closed interval [start, end].\n"
        << "  --seed <n>               Override kv.seed.\n"
        << "  --value-size <bytes>     Override kv.value_size.\n"
        << "  --batch-size <n>         Override bench.batch_size.\n"
        << "  --timeout <ms>           Override default wait timeout.\n"
        << "  --check                  Run consistency check when supported.\n"
        << "  --output <path>          Override output.path.\n"
        << "\n"
        << "Bench options:\n"
        << "  --op <op>, --bench-op <op>, --io-size <bytes>, --concurrency <n>,\n"
        << "  --duration <sec>, --count <n>, --warmup <sec>, --read-ratio <n>,\n"
        << "  --write-ratio <n>,\n"
        << "  --progress               Print one benchmark progress line per second.\n\n"
        << "Examples:\n"
        << "  export KV_TEST_CONFIG=/abs/path/to/asu_kv_test.conf\n"
        << "  kv-test connect\n"
        << "  kv-test store --key hello --check\n"
        << "  kv-test retrieve --keys hello,world --check\n"
        << "  kv-test delete --keys-file ./keys.txt\n"
        << "  kv-test exist --prefix user- --key-start 100 --key-end 199\n";
}

void PrintCommandHelp(CommandType command)
{
    switch (command) {
        case CommandType::CONNECT:
            std::cout << "Usage: kv-test connect [--configpath <path>] [--timeout <ms>]\n";
            break;
        case CommandType::CONFIG_CHECK:
            std::cout << "Usage: kv-test config check [--configpath <path>]\n";
            break;
        case CommandType::VERSION:
            std::cout << "Usage: kv-test version\n       kv-test --version\n";
            break;
        case CommandType::STORE:
            std::cout << "Usage: kv-test store (--key <key>|--keys <list>|--keys-file <path>|"
                         "--count <n>|--prefix <p> --key-start <n> --key-end <n>) "
                         "[--value-size <bytes>] [--seed <n>] [--check]\n";
            break;
        case CommandType::RETRIEVE:
            std::cout << "Usage: kv-test retrieve (--key <key>|--keys <list>|"
                         "--keys-file <path>|--count <n>|--prefix <p> --key-start <n> "
                         "--key-end <n>) "
                         "[--value-size <bytes>] [--seed <n>] [--check]\n";
            break;
        case CommandType::DELETE:
            std::cout << "Usage: kv-test delete (--key <key>|--keys <list>|--keys-file <path>|"
                         "--count <n>|--prefix <p> --key-start <n> --key-end <n>) "
                         "[--seed <n>] [--check]\n";
            break;
        case CommandType::EXIST:
            std::cout << "Usage: kv-test exist (--key <key>|--keys <list>|--keys-file <path>|"
                         "--count <n>|--prefix <p> --key-start <n> --key-end <n>) "
                         "[--seed <n>]\n";
            break;
        case CommandType::BATCH_STORE:
            std::cout << "Usage: kv-test batch-store (--keys <list>|--keys-file <path>|"
                         "--count <n>|--prefix <p> --key-start <n> --key-end <n>) "
                         "[--batch-size <n>] [--value-size <bytes>] [--check]\n";
            break;
        case CommandType::BATCH_RETRIEVE:
            std::cout << "Usage: kv-test batch-retrieve (--keys <list>|--keys-file <path>|"
                         "--count <n>|--prefix <p> --key-start <n> --key-end <n>) "
                         "[--batch-size <n>] [--value-size <bytes>] [--check]\n";
            break;
        case CommandType::POWER_CYCLE_PREPARE:
            std::cout << "Usage: kv-test power-cycle prepare (--keys <list>|--keys-file <path>|"
                         "--count <n>|--prefix <p> --key-start <n> --key-end <n>) "
                         "[--value-size <bytes>] [--seed <n>]\n";
            break;
        case CommandType::POWER_CYCLE_VERIFY:
            std::cout << "Usage: kv-test power-cycle verify (--keys <list>|--keys-file <path>|"
                         "--count <n>|--prefix <p> --key-start <n> --key-end <n>) "
                         "[--value-size <bytes>] [--seed <n>] [--check]\n";
            break;
        case CommandType::BENCH:
            std::cout << "Usage: kv-test bench [store|retrieve|batch-store|batch-retrieve|mix] "
                         "[--io-size <bytes>] [--concurrency <n>] "
                         "[--duration <sec>] [--count <n>]\n"
                      << "       kv-test bench --op <op> [--batch-size <n>] [--warmup <sec>] "
                         "[--read-ratio <n>] [--write-ratio <n>] [--progress]\n"
                      << "       A positive --count runs exactly that many io-size logical IOs "
                         "and overrides duration.\n";
            break;
        case CommandType::UNKNOWN:
        default: PrintGeneralHelp(); break;
    }
}

std::string LoadVersion()
{
    std::ifstream versionFile{"version.ini"};
    std::string line;
    while (std::getline(versionFile, line)) {
        const std::string prefix = "VLLM_UC_VERSION=";
        if (line.rfind(prefix, 0) == 0) { return line.substr(prefix.size()); }
    }
    return "unknown";
}

void PrintVersion() { std::cout << "kv-test version " << LoadVersion() << '\n'; }

void PrintFailure(const Status& status)
{
    std::cerr << kAnsiRed << "kv-test: failed" << kAnsiReset;
    if (!status.message.empty()) { std::cerr << ": " << status.message; }
    std::cerr << " (exit_code=" << ToExitCode(status) << ")\n";
}

void PrintExistSummary(const CommandOptions& options, const CommandResult& result)
{
    if (options.command != CommandType::EXIST) { return; }

    if (options.singleKeyRequested && options.keys.size() == 1 &&
        result.queryResult.exists.size() == 1) {
        std::cout << "exist: key=" << options.keys.front()
                  << " result=" << (result.queryResult.exists.front() != 0 ? "exists" : "missing")
                  << '\n';
        return;
    }

    const auto existsCount = static_cast<std::uint64_t>(
        std::count_if(result.queryResult.exists.begin(), result.queryResult.exists.end(),
                      [](std::uint8_t exists) { return exists != 0; }));
    const auto total = static_cast<std::uint64_t>(result.queryResult.exists.size());
    std::cout << "exist: total=" << total << " exists=" << existsCount
              << " missing=" << (total - existsCount) << '\n';
}

void PrintConsistencySummary(const CommandResult& result)
{
    const auto& summary = result.consistency;
    if (!summary.enabled) { return; }

    std::cout << "check: checked=" << summary.checked << " passed=" << summary.passed
              << " failed=" << summary.failed;
    if (!summary.key.empty()) {
        std::cout << " key=" << summary.key << " expected=" << summary.expected
                  << " actual=" << summary.actual;
    }
    std::cout << '\n';
}

void PrintBenchSummary(const CommandOptions& options, const CommandResult& result)
{
    if (options.command != CommandType::BENCH) { return; }

    const auto& metrics = result.benchMetrics;
    std::cout << "bench: op=" << BenchOpTypeName(metrics.op) << "\nelapsed_sec=" << std::fixed
              << std::setprecision(3) << metrics.elapsedSec
              << "\noperations=" << metrics.completedOperations
              << "\nentries=" << metrics.completedEntries << "\nbytes=" << metrics.completedBytes
              << "\nio_count=" << options.ioCount
              << "\nbandwidth_mib_s=" << FormatMiBPerSec(metrics.avgBandwidthBytesPerSec)
              << "\niops=" << std::fixed << std::setprecision(2) << metrics.avgIops
              << "\nbatch_iops=" << metrics.avgBatchIops
              << "\nlatency_avg_us=" << metrics.latency.avgUs
              << "\nlatency_p99_9_us=" << metrics.latency.p99_9Us
              << "\nerrors=" << metrics.errorCount << '\n';
}

void PrintSuccess(const CommandOptions& options, const CommandResult& result)
{
    std::cout << kAnsiGreen << "kv-test: succeeded" << kAnsiReset
              << "\ncommand=" << CommandTypeName(options.command)
              << "\nconfig=" << options.configPath << '\n';
    PrintExistSummary(options, result);
    PrintConsistencySummary(result);
    PrintBenchSummary(options, result);
}

Status CreateClient(std::unique_ptr<kv::AsuClient>& client)
{
    Status status;
    client = AsuRuntimeProxy::Instance().CreateAsuClient(nullptr, status);
    return status;
}

PayloadBufferPlacement PayloadPlacementForConfig(const KvTestConfig& config)
{
    return UsesDevicePayloadBuffers(config) ? PayloadBufferPlacement::ASCEND_DEVICE
                                            : PayloadBufferPlacement::HOST;
}

}  // namespace

KvTestApp::KvTestApp() = default;

int KvTestApp::Run(int argc, char** argv)
{
    CommandOptions options;
    auto status = argParser_.Parse(argc, argv, options);
    if (!status.Ok()) {
        PrintFailure(status);
        return ToExitCode(status);
    }
    if (options.helpRequested) {
        if (options.command == CommandType::UNKNOWN) {
            PrintGeneralHelp();
        } else {
            PrintCommandHelp(options.command);
        }
        return kExitSuccess;
    }
    if (options.command == CommandType::VERSION) {
        PrintVersion();
        return kExitSuccess;
    }

    status = configLoader_.ResolveConfigPath(options.configPath, options.configPath);
    if (!status.Ok()) {
        PrintFailure(status);
        return ToExitCode(status);
    }

    KvTestConfig config;
    status = configLoader_.Load(options.configPath, config);
    if (!status.Ok()) {
        PrintFailure(status);
        return ToExitCode(status);
    }

    status = configLoader_.MergeCommandOptions(options, config);
    if (!status.Ok()) {
        PrintFailure(status);
        return ToExitCode(status);
    }

    MaybePrepareFakeBackend(config);
    const auto effectiveOptions = BuildEffectiveOptions(options, config);

    status = hcommConfigAdapter_.ValidateChannelSource(config);
    if (!status.Ok()) {
        PrintFailure(status);
        return ToExitCode(status);
    }

    if (options.command == CommandType::CONFIG_CHECK) {
        std::cout << kAnsiGreen << "kv-test: succeeded" << kAnsiReset
                  << " command=config check config=" << options.configPath << '\n';
        std::cout << "config: key_prefix=" << config.keyPrefix << " count=" << config.count
                  << " value_size=" << config.valueSize
                  << " wait_timeout_ms=" << config.asuClientConfig.defaultWaitTimeoutMs
                  << " output=" << config.output.path << '\n';
        return kExitSuccess;
    }

    PayloadBufferRuntime payloadBufferRuntime;
    status = payloadBufferRuntime.MaybeSetUp(config);
    if (!status.Ok()) {
        PrintFailure(status);
        return ToExitCode(status);
    }

    status = resultWriter_.Open(config.output);
    if (!status.Ok()) {
        PrintFailure(status);
        return ToExitCode(status);
    }

    CommandResult result;
    std::unique_ptr<kv::AsuClient> client;
    status = CreateClient(client);
    if (!status.Ok()) {
        PrintFailure(status);
        return ToExitCode(status);
    }
    AsuClientRunner clientRunner(std::move(client));
    status = clientRunner.Init(config);
    if (status.Ok()) { status = RunCommand(effectiveOptions, config, clientRunner, result); }

    auto shutdownStatus = clientRunner.Shutdown();
    if (status.Ok() && !shutdownStatus.Ok()) { status = shutdownStatus; }

    result.status = status;
    auto writeStatus = resultWriter_.WriteSummary(effectiveOptions, result);
    if (status.Ok() && !writeStatus.Ok()) { status = writeStatus; }

    auto closeStatus = resultWriter_.Close();
    if (status.Ok() && !closeStatus.Ok()) { status = closeStatus; }

    if (status.Ok()) {
        PrintSuccess(effectiveOptions, result);
    } else {
        PrintFailure(status);
        PrintConsistencySummary(result);
    }
    return ToExitCode(status);
}

Status KvTestApp::RunCommand(const CommandOptions& options, const KvTestConfig& config,
                             AsuClientRunner& clientRunner, CommandResult& result)
{
    switch (options.command) {
        case CommandType::CONNECT: return Status::Success();
        case CommandType::CONFIG_CHECK:
        case CommandType::VERSION: return Status::Success();
        case CommandType::STORE:
        case CommandType::BATCH_STORE:
        case CommandType::POWER_CYCLE_PREPARE:
            return RunStoreLikeCommand(options, config, clientRunner, result);
        case CommandType::RETRIEVE:
        case CommandType::BATCH_RETRIEVE:
        case CommandType::POWER_CYCLE_VERIFY:
            return RunRetrieveLikeCommand(options, config, clientRunner, result);
        case CommandType::DELETE: return RunDeleteCommand(options, config, clientRunner, result);
        case CommandType::EXIST: return RunExistCommand(options, config, clientRunner, result);
        case CommandType::BENCH: return benchRunner_.Run(options, config, clientRunner, result);
        case CommandType::UNKNOWN:
        default: return Status::Error(kExitInvalidArgument, "unknown kv-test command");
    }
}

Status KvTestApp::RunStoreLikeCommand(const CommandOptions& options, const KvTestConfig& config,
                                      AsuClientRunner& clientRunner, CommandResult& result)
{
    GeneratedData data;
    auto status = generator_.Generate(options, config, data);
    if (!status.Ok()) { return status; }

    const auto payloadPlacement = PayloadPlacementForConfig(config);
    const auto allocationPolicy = AllocationPolicyForConfig(config);
    const auto logicalDeviceId = ResolvePayloadDeviceId(config);
    BufferSet buffers;
    status = bufferAllocator_.BuildStoreBuffers(data, payloadPlacement, allocationPolicy,
                                                logicalDeviceId, buffers);
    if (!status.Ok()) { return status; }

    status = clientRunner.RegisterBuffers(buffers);
    if (!status.Ok()) {
        auto unregisterStatus = clientRunner.UnregisterBuffers(buffers);
        if (unregisterStatus.Ok()) { return status; }
        return unregisterStatus;
    }

    const SubmitMode submitMode = options.command == CommandType::STORE
                                      ? SubmitMode::SINGLE_ENTRY_PER_CALL
                                      : SubmitMode::ALL_ENTRIES_IN_ONE_CALL;
    status = clientRunner.Store(buffers, submitMode, options.timeoutMs, result);

    auto unregisterStatus = clientRunner.UnregisterBuffers(buffers);
    if (status.Ok() && !unregisterStatus.Ok()) { status = unregisterStatus; }
    if (status.Ok() && options.command == CommandType::POWER_CYCLE_PREPARE) {
        status = WritePowerCycleMetadata(options, config);
    }
    if (!status.Ok() || !options.check) { return status; }

    BufferSet retrievedBuffers;
    status = bufferAllocator_.BuildRetrieveBuffers(data, payloadPlacement, allocationPolicy,
                                                   logicalDeviceId, retrievedBuffers);
    if (!status.Ok()) { return status; }

    status = clientRunner.RegisterBuffers(retrievedBuffers);
    if (!status.Ok()) {
        auto retrieveUnregisterStatus = clientRunner.UnregisterBuffers(retrievedBuffers);
        if (retrieveUnregisterStatus.Ok()) { return status; }
        return retrieveUnregisterStatus;
    }

    CommandResult retrieveResult;
    status = clientRunner.Retrieve(retrievedBuffers, submitMode, options.timeoutMs, retrieveResult);
    if (status.Ok()) { status = bufferAllocator_.CopyDeviceBuffersToHost(retrievedBuffers); }
    if (status.Ok()) {
        status = consistencyChecker_.CheckStoreResult(data, retrievedBuffers, retrieveResult,
                                                      result.consistency);
    }

    auto retrieveUnregisterStatus = clientRunner.UnregisterBuffers(retrievedBuffers);
    if (status.Ok() && !retrieveUnregisterStatus.Ok()) { status = retrieveUnregisterStatus; }
    return status;
}

Status KvTestApp::RunRetrieveLikeCommand(const CommandOptions& options, const KvTestConfig& config,
                                         AsuClientRunner& clientRunner, CommandResult& result)
{
    GeneratedData data;
    auto status = generator_.Generate(options, config, data);
    if (!status.Ok()) { return status; }
    if (options.command == CommandType::POWER_CYCLE_VERIFY) {
        status = ValidatePowerCycleMetadata(options, config);
        if (!status.Ok()) { return status; }
    }

    const auto payloadPlacement = PayloadPlacementForConfig(config);
    const auto allocationPolicy = AllocationPolicyForConfig(config);
    const auto logicalDeviceId = ResolvePayloadDeviceId(config);
    BufferSet buffers;
    status = bufferAllocator_.BuildRetrieveBuffers(data, payloadPlacement, allocationPolicy,
                                                   logicalDeviceId, buffers);
    if (!status.Ok()) { return status; }

    status = clientRunner.RegisterBuffers(buffers);
    if (!status.Ok()) {
        auto unregisterStatus = clientRunner.UnregisterBuffers(buffers);
        if (unregisterStatus.Ok()) { return status; }
        return unregisterStatus;
    }

    const SubmitMode submitMode = options.command == CommandType::RETRIEVE
                                      ? SubmitMode::SINGLE_ENTRY_PER_CALL
                                      : SubmitMode::ALL_ENTRIES_IN_ONE_CALL;
    status = clientRunner.Retrieve(buffers, submitMode, options.timeoutMs, result);
    const bool checkResult = options.check || options.command == CommandType::POWER_CYCLE_VERIFY;
    if (status.Ok() && checkResult) { status = bufferAllocator_.CopyDeviceBuffersToHost(buffers); }
    if (status.Ok() && checkResult) {
        status = consistencyChecker_.CheckRetrieveResult(data, buffers, result, result.consistency);
    }

    auto unregisterStatus = clientRunner.UnregisterBuffers(buffers);
    if (status.Ok() && !unregisterStatus.Ok()) { status = unregisterStatus; }
    return status;
}

Status KvTestApp::RunDeleteCommand(const CommandOptions& options, const KvTestConfig& config,
                                   AsuClientRunner& clientRunner, CommandResult& result)
{
    GeneratedData data;
    auto status = generator_.Generate(options, config, data);
    if (!status.Ok()) { return status; }

    status = clientRunner.Delete(data.keys, options.timeoutMs, result);
    if (!status.Ok() || !options.check) { return status; }

    CommandResult existResult;
    status = clientRunner.Exist(data.keys, options.timeoutMs, existResult);
    if (!status.Ok()) { return status; }
    return consistencyChecker_.CheckDeleteResult(data.keys, result, existResult,
                                                 result.consistency);
}

Status KvTestApp::RunExistCommand(const CommandOptions& options, const KvTestConfig& config,
                                  AsuClientRunner& clientRunner, CommandResult& result)
{
    GeneratedData data;
    auto status = generator_.Generate(options, config, data);
    if (!status.Ok()) { return status; }

    return clientRunner.Exist(data.keys, options.timeoutMs, result);
}

}  // namespace kv::bench
