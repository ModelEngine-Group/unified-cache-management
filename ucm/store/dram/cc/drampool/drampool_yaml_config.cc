/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "drampool_config.h"
#include "parse_utils.h"

namespace UC::DramPool {
namespace {

using Dram::ParseBool;
using Dram::ParseDouble;
using Dram::ParseInt32;
using Dram::ParseUint16;
using Dram::ParseUint32;
using Dram::ParseUint64;
using Dram::ToLower;
using Dram::Trim;

struct YamlSection {
    std::size_t indent{0};
    std::string key;
};

struct EndpointEntry {
    std::string twoSided;
    std::string oneSided;
    bool hasTwoSided{false};
    bool hasOneSided{false};
};

constexpr const char* kRequiredRuntimeConfigKeys[] = {
    "transport.device_ids",
    "queue.request_depth",
    "queue.completion_depth",
    "request_receiver.idle_wait_us",
    "poller.pending_depth",
    "flag_buffer.capacity_mb",
    "flag_buffer.slot_size_bytes",
    "gc.enabled",
    "gc.interval_ms",
    "metadata.periodic_eviction_policy",
    "metadata.deep_eviction_policy",
    "metadata.lease_time_ms",
    "metadata.default_evict_ratio",
    "metadata.evict_period_ms",
    "operation.timeout_ms",
    "logger.level",
    "logger.dir",
    "logger.max_files",
    "logger.max_size_mb",
};

std::string BuildSectionPath(const std::vector<YamlSection>& sections)
{
    std::string path;
    for (const auto& section : sections) {
        if (!path.empty()) { path += '.'; }
        path += section.key;
    }
    return path;
}

std::string StripYamlComment(const std::string& line)
{
    char quote = '\0';
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (quote != '\0') {
            if (character == quote) { quote = '\0'; }
            continue;
        }
        if (character == '\'' || character == '"') {
            quote = character;
        } else if (character == '#') {
            return line.substr(0, index);
        }
    }
    return line;
}

Status ParseYamlScalar(const std::string& key, std::string value, std::string& output)
{
    value = Trim(value);
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                              (value.front() == '\'' && value.back() == '\''))) {
        value = value.substr(1, value.size() - 2);
    } else if ((!value.empty() && (value.front() == '"' || value.front() == '\'')) ||
               (!value.empty() && (value.back() == '"' || value.back() == '\''))) {
        return Status::InvalidParam("unmatched quote in YAML key {}", key);
    }
    output = std::move(value);
    return Status::OK();
}

Status ParseEvictionPolicyValue(const std::string& key, const std::string& value,
                                EvictionPolicyType& output)
{
    const auto normalized = ToLower(value);
    if (normalized == "ttl") {
        output = EvictionPolicyType::TTL;
        return Status::OK();
    }
    if (normalized == "position") {
        output = EvictionPolicyType::POSITION;
        return Status::OK();
    }
    return Status::InvalidParam("unsupported eviction policy for {}: {}", key, value);
}

Status ApplyEndpointEntryValue(EndpointEntry& entry, const std::string& key,
                               const std::string& value, std::uint32_t lineNumber)
{
    if (key == "two_sided") {
        if (entry.hasTwoSided) {
            return Status::InvalidParam("duplicate transport endpoint two_sided at line {}",
                                        lineNumber);
        }
        entry.twoSided = value;
        entry.hasTwoSided = true;
        return Status::OK();
    }
    if (key == "one_sided") {
        if (entry.hasOneSided) {
            return Status::InvalidParam("duplicate transport endpoint one_sided at line {}",
                                        lineNumber);
        }
        entry.oneSided = value;
        entry.hasOneSided = true;
        return Status::OK();
    }
    return Status::InvalidParam("unknown transport endpoint key at line {}: {}", lineNumber, key);
}

Status CommitEndpointEntry(EndpointEntry& entry, DramPoolConfig& config,
                           std::unordered_set<std::string>& oneSidedIds)
{
    if (!entry.hasTwoSided || !entry.hasOneSided) {
        return Status::InvalidParam("each transport endpoint requires two_sided and one_sided");
    }
    transport::Endpoint twoSided;
    transport::Endpoint oneSided;
    if (auto status =
            ParseDramPoolEndpoint("transport.endpoints.two_sided", entry.twoSided, twoSided);
        status.Failure()) {
        return status;
    }
    if (auto status =
            ParseDramPoolEndpoint("transport.endpoints.one_sided", entry.oneSided, oneSided);
        status.Failure()) {
        return status;
    }
    const auto twoSidedId = twoSided.ToString();
    const auto oneSidedId = oneSided.ToString();
    if (!config.twoSidedToOneSided.emplace(twoSidedId, oneSidedId).second) {
        return Status::InvalidParam("duplicate transport two_sided endpoint: {}", twoSidedId);
    }
    if (!oneSidedIds.insert(oneSidedId).second) {
        return Status::InvalidParam("duplicate transport one_sided endpoint: {}", oneSidedId);
    }
    entry = EndpointEntry{};
    return Status::OK();
}

using RuntimeConfigParser = std::function<Status(DramPoolConfig&, const std::string&)>;

template <typename T>
RuntimeConfigParser BindConfigParser(Status (*parser)(const std::string&, T&),
                                     T DramPoolConfig::* member)
{
    return [parser, member](DramPoolConfig& config, const std::string& value) {
        return parser(value, config.*member);
    };
}

template <typename T>
RuntimeConfigParser BindConfigParser(Status (*parser)(const std::string&, const std::string&, T&),
                                     std::string key, T DramPoolConfig::* member)
{
    return
        [parser, key = std::move(key), member](DramPoolConfig& config, const std::string& value) {
            return parser(key, value, config.*member);
        };
}

Status ParseStringValue(const std::string& value, std::string& output)
{
    output = value;
    return Status::OK();
}

Status ParseLowerStringValue(const std::string& value, std::string& output)
{
    output = ToLower(value);
    return Status::OK();
}

const std::unordered_map<std::string_view, RuntimeConfigParser>& GetRuntimeConfigParsers()
{
    static const std::unordered_map<std::string_view, RuntimeConfigParser> parsers = {
        {"health.port", BindConfigParser(ParseUint16, &DramPoolConfig::healthPort)},
        {"queue.request_depth", BindConfigParser(ParseUint32, &DramPoolConfig::requestQueueDepth)},
        {"queue.completion_depth",
         BindConfigParser(ParseUint32, &DramPoolConfig::completionQueueDepth)},
        {"request_receiver.idle_wait_us",
         BindConfigParser(ParseUint32, &DramPoolConfig::requestReceiverIdleWaitUs)},
        {"poller.pending_depth",
         BindConfigParser(ParseUint32, &DramPoolConfig::pollerPendingDepth)},
        {"flag_buffer.capacity_mb",
         BindConfigParser(ParseUint64, &DramPoolConfig::flagBufferCapacityMb)},
        {"flag_buffer.slot_size_bytes",
         BindConfigParser(ParseUint64, &DramPoolConfig::flagBufferSlotSizeBytes)},
        {"gc.enabled", BindConfigParser(ParseBool, &DramPoolConfig::gcEnabled)},
        {"gc.interval_ms", BindConfigParser(ParseUint32, &DramPoolConfig::gcIntervalMs)},
        {"metadata.periodic_eviction_policy",
         BindConfigParser(ParseEvictionPolicyValue, "metadata.periodic_eviction_policy",
         &DramPoolConfig::metadataPeriodicEvictionPolicy)},
        {"metadata.deep_eviction_policy",
         BindConfigParser(ParseEvictionPolicyValue, "metadata.deep_eviction_policy",
         &DramPoolConfig::metadataDeepEvictionPolicy)},
        {"metadata.lease_time_ms",
         BindConfigParser(ParseUint64, &DramPoolConfig::metadataLeaseTimeMs)},
        {"metadata.default_evict_ratio",
         BindConfigParser(ParseDouble, &DramPoolConfig::metadataDefaultEvictRatio)},
        {"metadata.evict_period_ms",
         BindConfigParser(ParseUint64, &DramPoolConfig::metadataEvictPeriodMs)},
        {"operation.timeout_ms", BindConfigParser(ParseUint32, &DramPoolConfig::opTimeoutMs)},
        {"logger.level", BindConfigParser(ParseLowerStringValue, &DramPoolConfig::logLevel)},
        {"logger.dir", BindConfigParser(ParseStringValue, &DramPoolConfig::logDir)},
        {"logger.max_files", BindConfigParser(ParseUint32, &DramPoolConfig::logMaxFiles)},
        {"logger.max_size_mb", BindConfigParser(ParseUint32, &DramPoolConfig::logMaxSizeMb)}
    };
    return parsers;
}

Status ApplyRuntimeConfigValue(DramPoolConfig& config, const std::string& key,
                               const std::string& value)
{
    const auto& parsers = GetRuntimeConfigParsers();
    const auto parser = parsers.find(std::string_view{key});
    if (parser == parsers.end()) {
        return Status::InvalidParam("unknown DramPool runtime YAML key: {}", key);
    }
    return parser->second(config, value);
}

Status ValidateRuntimeConfig(DramPoolConfig& config)
{
    if (config.twoSidedToOneSided.empty()) {
        return Status::InvalidParam("transport.endpoints must not be empty");
    }
    const auto localControlId = config.addr.ToString();
    const auto localEndpoint = config.twoSidedToOneSided.find(localControlId);
    if (localEndpoint == config.twoSidedToOneSided.end()) {
        return Status::InvalidParam("transport.endpoints has no two_sided entry for --addr {}",
                                    localControlId);
    }
    for (const auto& endpoint : config.twoSidedToOneSided) {
        if (config.twoSidedToOneSided.find(endpoint.second) != config.twoSidedToOneSided.end()) {
            return Status::InvalidParam(
                "transport endpoint cannot be both two_sided and one_sided: {}", endpoint.second);
        }
    }
    if (config.transportDeviceIds.empty()) {
        return Status::InvalidParam("transport.device_ids must not be empty");
    }
    std::unordered_set<std::int32_t> deviceIds;
    for (const auto deviceId : config.transportDeviceIds) {
        if (deviceId < 0) {
            return Status::InvalidParam("transport.device_ids must not contain negative values");
        }
        if (!deviceIds.insert(deviceId).second) {
            return Status::InvalidParam("transport.device_ids contains duplicate device ID: {}",
                                        deviceId);
        }
    }
    if (config.requestQueueDepth < 2 || config.completionQueueDepth < 2) {
        return Status::InvalidParam("queue depths must be at least 2");
    }
    if (config.requestReceiverIdleWaitUs == 0) {
        return Status::InvalidParam("request_receiver.idle_wait_us must be greater than zero");
    }
    if (config.pollerPendingDepth == 0) {
        return Status::InvalidParam("poller.pending_depth must be greater than zero");
    }
    if (config.flagBufferCapacityMb == 0 || config.flagBufferSlotSizeBytes == 0) {
        return Status::InvalidParam(
            "flag_buffer.capacity_mb and flag_buffer.slot_size_bytes must be greater than zero");
    }
    if (config.flagBufferCapacityMb > std::numeric_limits<std::uint64_t>::max() / kBytesPerMiB) {
        return Status::InvalidParam("flag_buffer.capacity_mb is too large");
    }
    const auto capacityBytes = config.flagBufferCapacityMb * kBytesPerMiB;
    if (capacityBytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        config.flagBufferSlotSizeBytes >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        config.flagBufferSlotSizeBytes >
            std::numeric_limits<std::uint64_t>::max() - (kFlagBufferSlotAlignment - 1)) {
        return Status::InvalidParam("flag_buffer layout exceeds addressable process memory");
    }
    const auto slotStride = (config.flagBufferSlotSizeBytes + kFlagBufferSlotAlignment - 1) /
                            kFlagBufferSlotAlignment * kFlagBufferSlotAlignment;
    const auto slotCount = capacityBytes / slotStride;
    if (slotCount < config.pollerPendingDepth ||
        slotCount >= std::numeric_limits<std::uint32_t>::max()) {
        return Status::InvalidParam(
            "flag_buffer capacity must provide at least poller.pending_depth slots and fit "
            "the BufferPool index range");
    }
    config.flagBufferSlotCount = static_cast<std::uint32_t>(slotCount);
    if (config.gcEnabled && config.gcIntervalMs == 0) {
        return Status::InvalidParam("gc.interval_ms must be greater than zero when GC is enabled");
    }
    if (config.metadataLeaseTimeMs == 0) {
        return Status::InvalidParam("metadata.lease_time_ms must be greater than zero");
    }
    if (config.metadataLeaseTimeMs >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return Status::InvalidParam("metadata.lease_time_ms is too large");
    }
    if (config.metadataDefaultEvictRatio < 0.0 || config.metadataDefaultEvictRatio > 1.0) {
        return Status::InvalidParam("metadata.default_evict_ratio must be in [0, 1]");
    }
    if (config.metadataEvictPeriodMs == 0) {
        return Status::InvalidParam("metadata.evict_period_ms must be greater than zero");
    }
    if (config.metadataEvictPeriodMs >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return Status::InvalidParam("metadata.evict_period_ms is too large");
    }
    if (config.opTimeoutMs == 0) {
        return Status::InvalidParam("operation.timeout_ms must be greater than zero");
    }
    if (config.logLevel != "trace" && config.logLevel != "debug" && config.logLevel != "info" &&
        config.logLevel != "warn" && config.logLevel != "error" && config.logLevel != "critical") {
        return Status::InvalidParam("unsupported logger.level: {}", config.logLevel);
    }
    if (Trim(config.logDir).empty()) {
        return Status::InvalidParam("logger.dir must not be empty");
    }
    if (config.logMaxFiles == 0 || config.logMaxSizeMb == 0 ||
        config.logMaxFiles > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        config.logMaxSizeMb > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        return Status::InvalidParam("logger rotation values must fit positive int values");
    }
    return Status::OK();
}

Status ValidateRequiredRuntimeConfigKeys(const std::unordered_set<std::string>& configuredKeys)
{
    for (const char* key : kRequiredRuntimeConfigKeys) {
        if (configuredKeys.find(key) == configuredKeys.end()) {
            return Status::InvalidParam("DramPool runtime YAML is missing required key: {}", key);
        }
    }
    return Status::OK();
}

}  // namespace

Status ParseYamlConfig(const std::string& path, DramPoolConfig& config)
{
    std::ifstream input{path};
    if (!input.is_open()) {
        return Status::Error("failed to open DramPool runtime YAML, path=" + path);
    }

    // Keep the caller's launch configuration intact if YAML parsing fails.
    DramPoolConfig loadedConfig = config;
    loadedConfig.transportDeviceIds.clear();
    loadedConfig.twoSidedToOneSided.clear();
    std::vector<YamlSection> sections;
    std::unordered_set<std::string> configuredKeys;
    std::unordered_set<std::string> oneSidedIds;
    EndpointEntry endpointEntry;
    bool endpointEntryActive = false;
    bool endpointsSectionSeen = false;
    std::string line;
    std::uint32_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        line = StripYamlComment(line);
        if (Trim(line).empty()) { continue; }
        if (line.find('\t') != std::string::npos) {
            return Status::InvalidParam("YAML line {} uses a tab for indentation", lineNumber);
        }

        const auto firstContent = line.find_first_not_of(' ');
        const auto indent = firstContent == std::string::npos ? 0 : firstContent;
        auto content = Trim(line.substr(indent));
        while (!sections.empty() && sections.back().indent >= indent) { sections.pop_back(); }
        auto sectionPath = BuildSectionPath(sections);
        if (endpointEntryActive && sectionPath != "transport.endpoints") {
            if (auto status = CommitEndpointEntry(endpointEntry, loadedConfig, oneSidedIds);
                status.Failure()) {
                return status;
            }
            endpointEntryActive = false;
        }

        if (content.rfind("- ", 0) == 0) {
            if (sectionPath == "transport.device_ids") {
                auto itemToken = Trim(content.substr(2));
                if (itemToken.empty()) {
                    return Status::InvalidParam("empty transport.device_ids entry at line {}",
                                                lineNumber);
                }
                std::string itemValue;
                auto status = ParseYamlScalar("transport.device_ids", itemToken, itemValue);
                if (status.Failure()) { return status; }
                std::int32_t deviceId = 0;
                status = ParseInt32(itemValue, deviceId);
                if (status.Failure()) { return status; }
                loadedConfig.transportDeviceIds.push_back(deviceId);
                continue;
            }
            if (sectionPath != "transport.endpoints") {
                return Status::InvalidParam("YAML sequence is unsupported at line {}", lineNumber);
            }
            if (endpointEntryActive) {
                if (auto status = CommitEndpointEntry(endpointEntry, loadedConfig, oneSidedIds);
                    status.Failure()) {
                    return status;
                }
            }
            endpointEntryActive = true;
            endpointsSectionSeen = true;
            content = Trim(content.substr(2));
        }

        const auto separator = content.find(':');
        if (separator == std::string::npos || separator == 0) {
            return Status::InvalidParam("invalid YAML mapping at line {}", lineNumber);
        }

        const auto key = Trim(content.substr(0, separator));
        if (key.empty() || key.find_first_of(" \t") != std::string::npos) {
            return Status::InvalidParam("invalid YAML key at line {}", lineNumber);
        }
        const auto scalarToken = content.substr(separator + 1);
        const bool hasScalar = !Trim(scalarToken).empty();
        std::string value;
        auto status = ParseYamlScalar(key, scalarToken, value);
        if (status.Failure()) { return status; }

        if (sectionPath == "transport.endpoints") {
            if (!endpointEntryActive || !hasScalar) {
                return Status::InvalidParam("invalid transport endpoint at line {}", lineNumber);
            }
            status = ApplyEndpointEntryValue(endpointEntry, key, value, lineNumber);
            if (status.Failure()) { return status; }
            continue;
        }
        if (!hasScalar) {
            sections.push_back(YamlSection{indent, key});
            const auto newSectionPath = BuildSectionPath(sections);
            if (newSectionPath == "transport.device_ids" &&
                !configuredKeys.insert(newSectionPath).second) {
                return Status::InvalidParam("duplicate YAML key: {}", newSectionPath);
            }
            if (newSectionPath == "transport.endpoints") { endpointsSectionSeen = true; }
            continue;
        }

        std::string fullKey;
        for (const auto& section : sections) {
            if (!fullKey.empty()) { fullKey += '.'; }
            fullKey += section.key;
        }
        if (!fullKey.empty()) { fullKey += '.'; }
        fullKey += key;
        if (!configuredKeys.insert(fullKey).second) {
            return Status::InvalidParam("duplicate YAML key: {}", fullKey);
        }
        status = ApplyRuntimeConfigValue(loadedConfig, fullKey, value);
        if (status.Failure()) { return status; }
    }
    if (endpointEntryActive) {
        if (auto status = CommitEndpointEntry(endpointEntry, loadedConfig, oneSidedIds);
            status.Failure()) {
            return status;
        }
    }
    if (!endpointsSectionSeen) {
        return Status::InvalidParam("DramPool runtime YAML is missing transport.endpoints");
    }
    if (const auto status = ValidateRequiredRuntimeConfigKeys(configuredKeys); status.Failure()) {
        return status;
    }
    if (const auto status = ValidateRuntimeConfig(loadedConfig); status.Failure()) {
        return status;
    }
    config = std::move(loadedConfig);
    return Status::OK();
}

}  // namespace UC::DramPool
