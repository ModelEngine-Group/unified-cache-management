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
#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "drampool_config.h"

namespace UC::DramPool {
namespace {

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
    "transport.device_id",
    "queue.request_depth",
    "queue.completion_depth",
    "request_receiver.idle_wait_us",
    "poller.pending_depth",
    "gc.enabled",
    "gc.interval_ms",
    "metadata.lease_time_ms",
    "metadata.default_evict_ratio",
    "metadata.evict_period_ms",
    "operation.timeout_ms",
    "logger.level",
    "logger.dir",
    "logger.max_files",
    "logger.max_size_mb",
};

std::string Trim(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) { return ""; }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

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

Status ParseUint32Value(const std::string& key, const std::string& value, std::uint32_t& output)
{
    try {
        if (value.empty() || value.front() == '-') {
            throw std::invalid_argument("expected an unsigned integer");
        }
        std::size_t parsed = 0;
        const auto number = std::stoull(value, &parsed, 0);
        if (parsed != value.size() || number > std::numeric_limits<std::uint32_t>::max()) {
            throw std::out_of_range("outside uint32 range");
        }
        output = static_cast<std::uint32_t>(number);
    } catch (const std::exception& error) {
        return Status::InvalidParam("invalid YAML value for {}: {}", key, error.what());
    }
    return Status::OK();
}

Status ParseUint64Value(const std::string& key, const std::string& value, std::uint64_t& output)
{
    try {
        if (value.empty() || value.front() == '-') {
            throw std::invalid_argument("expected an unsigned integer");
        }
        std::size_t parsed = 0;
        const auto number = std::stoull(value, &parsed, 0);
        if (parsed != value.size()) { throw std::invalid_argument("unexpected trailing data"); }
        output = number;
    } catch (const std::exception& error) {
        return Status::InvalidParam("invalid YAML value for {}: {}", key, error.what());
    }
    return Status::OK();
}

Status ParseDoubleValue(const std::string& key, const std::string& value, double& output)
{
    try {
        std::size_t parsed = 0;
        const auto number = std::stod(value, &parsed);
        if (parsed != value.size() || !std::isfinite(number)) {
            throw std::invalid_argument("expected a finite number");
        }
        output = number;
    } catch (const std::exception& error) {
        return Status::InvalidParam("invalid YAML value for {}: {}", key, error.what());
    }
    return Status::OK();
}

Status ParseInt32Value(const std::string& key, const std::string& value, std::int32_t& output)
{
    try {
        std::size_t parsed = 0;
        const auto number = std::stoll(value, &parsed, 0);
        if (parsed != value.size() || number < std::numeric_limits<std::int32_t>::min() ||
            number > std::numeric_limits<std::int32_t>::max()) {
            throw std::out_of_range("outside int32 range");
        }
        output = static_cast<std::int32_t>(number);
    } catch (const std::exception& error) {
        return Status::InvalidParam("invalid YAML value for {}: {}", key, error.what());
    }
    return Status::OK();
}

Status ParseBoolValue(const std::string& key, const std::string& value, bool& output)
{
    const auto normalized = ToLower(value);
    if (normalized == "true" || normalized == "yes" || normalized == "on" || normalized == "1") {
        output = true;
        return Status::OK();
    }
    if (normalized == "false" || normalized == "no" || normalized == "off" || normalized == "0") {
        output = false;
        return Status::OK();
    }
    return Status::InvalidParam("invalid YAML boolean for {}", key);
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

Status ApplyRuntimeConfigValue(DramPoolConfig& config, const std::string& key,
                               const std::string& value)
{
    if (key == "transport.device_id") {
        return ParseInt32Value(key, value, config.transportDeviceId);
    }
    if (key == "queue.request_depth") {
        return ParseUint32Value(key, value, config.requestQueueDepth);
    }
    if (key == "queue.completion_depth") {
        return ParseUint32Value(key, value, config.completionQueueDepth);
    }
    if (key == "request_receiver.idle_wait_us") {
        return ParseUint32Value(key, value, config.requestReceiverIdleWaitUs);
    }
    if (key == "poller.pending_depth") {
        return ParseUint32Value(key, value, config.pollerPendingDepth);
    }
    if (key == "gc.enabled") { return ParseBoolValue(key, value, config.gcEnabled); }
    if (key == "gc.interval_ms") { return ParseUint32Value(key, value, config.gcIntervalMs); }
    if (key == "metadata.lease_time_ms") {
        return ParseUint64Value(key, value, config.metadataLeaseTimeMs);
    }
    if (key == "metadata.default_evict_ratio") {
        return ParseDoubleValue(key, value, config.metadataDefaultEvictRatio);
    }
    if (key == "metadata.evict_period_ms") {
        return ParseUint64Value(key, value, config.metadataEvictPeriodMs);
    }
    if (key == "operation.timeout_ms") { return ParseUint32Value(key, value, config.opTimeoutMs); }
    if (key == "logger.level") {
        config.logLevel = ToLower(value);
        return Status::OK();
    }
    if (key == "logger.dir") {
        config.logDir = value;
        return Status::OK();
    }
    if (key == "logger.max_files") { return ParseUint32Value(key, value, config.logMaxFiles); }
    if (key == "logger.max_size_mb") { return ParseUint32Value(key, value, config.logMaxSizeMb); }
    return Status::InvalidParam("unknown DramPool runtime YAML key: {}", key);
}

Status ValidateRuntimeConfig(const DramPoolConfig& config)
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
    if (config.transportDeviceId < 0) {
        return Status::InvalidParam("transport.device_id must not be negative");
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
            if (BuildSectionPath(sections) == "transport.endpoints") {
                endpointsSectionSeen = true;
            }
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
