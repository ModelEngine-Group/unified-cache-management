#include "kv_test_config_loader.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include "asu_runtime_proxy.h"

namespace kv::bench {

namespace {

constexpr int kExitInvalidArgument = 1;
constexpr const char* kConfigPathEnvVar = "KV_TEST_CONFIG";
constexpr std::uint64_t kMaxBenchIoSizeBytes = 0xFFFFFF;

std::string Trim(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) { return ""; }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string NormalizeKey(std::string key)
{
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return key;
}

std::uint64_t ParseUint64(const std::string& value)
{
    std::size_t parsed{0};
    const auto result = std::stoull(value, &parsed, 0);
    if (parsed != value.size()) { throw std::invalid_argument("invalid unsigned integer"); }
    return result;
}

std::uint32_t ParseUint32(const std::string& value)
{
    const auto result = ParseUint64(value);
    if (result > std::numeric_limits<std::uint32_t>::max()) {
        throw std::out_of_range("uint32 overflow");
    }
    return static_cast<std::uint32_t>(result);
}

std::unordered_map<std::string, std::string> LoadKeyValueFile(const std::string& configPath,
                                                              Status& status)
{
    std::ifstream configFile{configPath};
    if (!configFile.is_open()) {
        status = Status::Error(kExitInvalidArgument,
                               "failed to open kv-test config, path=" + configPath);
        return {};
    }

    std::unordered_map<std::string, std::string> values;
    std::string line;
    while (std::getline(configFile, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') { continue; }

        const auto pos = line.find('=');
        if (pos == std::string::npos) { continue; }

        const auto key = NormalizeKey(Trim(line.substr(0, pos)));
        const auto value = Trim(line.substr(pos + 1));
        if (!key.empty()) { values[key] = value; }
    }

    status = Status::Success();
    return values;
}

bool GetStringValue(const std::unordered_map<std::string, std::string>& values,
                    const std::string& key, std::string& result)
{
    const auto iter = values.find(NormalizeKey(key));
    if (iter == values.end()) { return false; }
    result = iter->second;
    return true;
}

bool GetUint64Value(const std::unordered_map<std::string, std::string>& values,
                    const std::string& key, std::uint64_t& result)
{
    const auto iter = values.find(NormalizeKey(key));
    if (iter == values.end()) { return false; }
    result = ParseUint64(iter->second);
    return true;
}

bool GetUint32Value(const std::unordered_map<std::string, std::string>& values,
                    const std::string& key, std::uint32_t& result)
{
    const auto iter = values.find(NormalizeKey(key));
    if (iter == values.end()) { return false; }
    result = ParseUint32(iter->second);
    return true;
}

bool GetStringAny(const std::unordered_map<std::string, std::string>& values,
                  const std::vector<std::string>& keys, std::string& result)
{
    for (const auto& key : keys) {
        if (GetStringValue(values, key, result)) { return true; }
    }
    return false;
}

bool GetUint64Any(const std::unordered_map<std::string, std::string>& values,
                  const std::vector<std::string>& keys, std::uint64_t& result)
{
    for (const auto& key : keys) {
        if (GetUint64Value(values, key, result)) { return true; }
    }
    return false;
}

bool GetUint32Any(const std::unordered_map<std::string, std::string>& values,
                  const std::vector<std::string>& keys, std::uint32_t& result)
{
    for (const auto& key : keys) {
        if (GetUint32Value(values, key, result)) { return true; }
    }
    return false;
}

Status ToKvTestConfigStatus(const kv::Status& status)
{
    if (status.ok()) { return Status::Success(); }
    return Status::Error(kExitInvalidArgument,
                         "failed to load asu client config: " + status.message);
}

}  // namespace

Status KvTestConfigLoader::ResolveConfigPath(const std::string& configPath,
                                             std::string& resolvedPath) const
{
    if (!configPath.empty()) {
        resolvedPath = configPath;
        return Status::Success();
    }

    const char* envValue = std::getenv(kConfigPathEnvVar);
    if (envValue != nullptr && envValue[0] != '\0') {
        resolvedPath = envValue;
        return Status::Success();
    }

    return Status::Error(kExitInvalidArgument,
                         "missing config path: set KV_TEST_CONFIG or pass --configpath");
}

Status KvTestConfigLoader::Load(const std::string& configPath, KvTestConfig& config) const
{
    config.behavior = ToolBehaviorConfig{};
    config.hcommProtocolMapping = HcommProtocolMapping{};
    config.bench = BenchConfig{};
    config.output = OutputConfig{};
    config.fakeBackend = KvTestFakeBackendConfig{};
    config.asuRuntime = AsuRuntimeLibraryConfig{};
    config.keyPrefix.clear();
    config.seed = 0;
    config.valueSize = 0;
    config.count = 0;
    config.memoryMaxBytes = kDefaultMemoryMaxBytes;

    Status status;
    auto values = LoadKeyValueFile(configPath, status);
    if (!status.Ok()) { return status; }

    try {
        GetStringAny(
            values,
            {"asu.client_library_path", "asu.client.library_path", "asu_client.library_path"},
            config.asuRuntime.clientLibraryPath);
        GetStringAny(values,
                     {"asu.transport_library_path", "asu.transport.library_path",
                      "asu_transport.library_path"},
                     config.asuRuntime.transportLibraryPath);

        status = AsuRuntimeProxy::Instance().Load(config.asuRuntime);
        if (!status.Ok()) { return status; }

        auto asuStatus =
            AsuRuntimeProxy::Instance().LoadAsuClientConfig(configPath, config.asuClientConfig);
        status = ToKvTestConfigStatus(asuStatus);
        if (!status.Ok()) { return status; }

        GetStringAny(values, {"fake_backend.path", "fakebackend.path"},
                     config.fakeBackend.storePath);
        GetUint64Any(values,
                     {"fake_backend.latency_ms", "fakebackend.latency_ms", "fake_backend.latencyms",
                      "fakebackend.latencyms"},
                     config.fakeBackend.latencyMs);
        GetUint64Any(values, {"fake_backend.worker_threads", "fakebackend.worker_threads"},
                     config.fakeBackend.workerThreads);

        GetStringAny(values, {"kv.key_prefix"}, config.keyPrefix);

        GetUint64Any(values, {"kv.value_size"}, config.valueSize);
        GetUint64Any(values, {"kv.seed"}, config.seed);
        GetUint64Any(values, {"kv.count"}, config.count);

        GetUint64Any(values, {"limits.memory_max_bytes"}, config.memoryMaxBytes);

        GetUint64Any(values, {"bench.io_size"}, config.bench.ioSize);
        GetUint32Any(values, {"bench.concurrency"}, config.bench.concurrency);
        GetUint64Any(values, {"bench.duration_sec"}, config.bench.durationSec);
        GetUint64Any(values, {"bench.io_count"}, config.bench.ioCount);
        GetUint64Any(values, {"bench.warmup_sec"}, config.bench.warmupSec);
        GetUint32Any(values, {"bench.read_ratio"}, config.bench.readRatio);
        GetUint32Any(values, {"bench.write_ratio"}, config.bench.writeRatio);
        GetUint32Any(values, {"bench.batch_size"}, config.bench.batchSize);

        GetStringAny(values, {"output.path"}, config.output.path);
        GetUint64Any(values, {"output.realtime_file_max_bytes"},
                     config.output.realtimeFileMaxBytes);
    } catch (const std::exception& e) {
        return Status::Error(kExitInvalidArgument,
                             "invalid kv-test config value in " + configPath + ": " + e.what());
    }

    return Status::Success();
}

Status KvTestConfigLoader::MergeCommandOptions(const CommandOptions& options,
                                               KvTestConfig& config) const
{
    if (options.seed != 0) { config.seed = options.seed; }
    if (options.count != 0) { config.count = options.count; }
    if (options.valueSize != 0) {
        config.valueSize = options.valueSize;
        if (options.command == CommandType::BENCH) { config.bench.ioSize = options.valueSize; }
    }
    if (options.batchSize != 0) { config.bench.batchSize = options.batchSize; }
    if (options.timeoutMs != 0) {
        config.asuClientConfig.defaultWaitTimeoutMs = options.timeoutMs;
        config.asuClientConfig.timeoutMs = options.timeoutMs;
        for (auto& transportConfig : config.asuClientConfig.transportConfigs) {
            transportConfig.timeoutMs = options.timeoutMs;
        }
    }
    if (!options.outputPath.empty()) { config.output.path = options.outputPath; }

    if (options.benchOp != BenchOpType::UNKNOWN) { config.bench.op = options.benchOp; }
    if (options.concurrency != 0) { config.bench.concurrency = options.concurrency; }
    if (options.durationSec != 0) {
        config.bench.durationSec = options.durationSec;
        config.bench.ioCount = 0;
    }
    if (options.ioCount != 0) { config.bench.ioCount = options.ioCount; }
    if (options.warmupSec != 0) { config.bench.warmupSec = options.warmupSec; }
    if (options.readRatio != 0) { config.bench.readRatio = options.readRatio; }
    if (options.writeRatio != 0) { config.bench.writeRatio = options.writeRatio; }

    const bool hasExplicitKeys = !options.keys.empty() || !options.keysFile.empty() ||
                                 options.keyStartSet || options.keyEndSet;
    if (config.count != 0 && config.keyPrefix.empty() && !hasExplicitKeys) {
        return Status::Error(kExitInvalidArgument,
                             "kv.key_prefix is required when count-based key generation is used");
    }
    if (config.count != 0 && config.valueSize == 0 && options.command != CommandType::DELETE &&
        options.command != CommandType::EXIST) {
        return Status::Error(kExitInvalidArgument,
                             "kv.value_size is required when count-based value generation is used");
    }
    if (options.command == CommandType::BENCH && config.bench.ioSize == 0) {
        config.bench.ioSize = config.valueSize;
    }
    if (options.command == CommandType::BENCH && config.bench.ioSize > kMaxBenchIoSizeBytes) {
        return Status::Error(kExitInvalidArgument,
                             "bench.io_size exceeds protocol 24-bit length limit");
    }

    return Status::Success();
}

}  // namespace kv::bench
