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
#include "trans_config_utils.h"
#include <fstream>
#include <limits>
#include <utility>
#include "kv_transport.h"
#include "parser_common.h"
#include "logger.h"

namespace kv {

Status ValidateSqeRequestAttrs(const std::unordered_map<std::string, std::string>& attrs)
{
    const auto parseInteger = [&attrs](const std::string& name, auto maxValue,
                                       std::uint64_t& out) -> Status {
        auto iter = attrs.find(name);
        if (iter == attrs.end()) { return Status::Error(StatusCode::NOT_FOUND, ""); }
        try {
            out = std::stoull(iter->second, nullptr, 0);
            if (out > maxValue) {
                UC_ERROR("Validate SQE attr failed name={} value={} reason=exceeds_range", name,
                         iter->second);
                return Status::Error(StatusCode::INVALID_ARGUMENT, name + " exceeds valid range");
            }
        } catch (const std::exception&) {
            UC_ERROR("Validate SQE attr failed name={} value={} reason=invalid_integer", name,
                     iter->second);
            return Status::Error(StatusCode::INVALID_ARGUMENT, name + " is not a valid integer");
        }
        return Status::OK();
    };

    const auto validateInteger = [&](const std::string& name, auto maxValue) -> Status {
        std::uint64_t dummy;
        auto s = parseInteger(name, maxValue, dummy);
        if (s.code == StatusCode::NOT_FOUND) { return Status::OK(); }
        return s;
    };

    const auto validateRequiredPositiveInteger = [&](const std::string& name,
                                                     auto maxValue) -> Status {
        std::uint64_t parsed;
        auto s = parseInteger(name, maxValue, parsed);
        if (s.code == StatusCode::NOT_FOUND) {
            UC_ERROR("Validate SQE attr failed name={} reason=missing_required", name);
            return Status::Error(StatusCode::INVALID_ARGUMENT, name + " is required");
        }
        if (!s.ok()) { return s; }
        if (parsed == 0) {
            UC_ERROR("Validate SQE attr failed name={} reason=must_be_positive", name);
            return Status::Error(StatusCode::INVALID_ARGUMENT, name + " must be greater than zero");
        }
        return Status::OK();
    };

    const auto validateBool = [&attrs](const std::string& name) -> Status {
        auto iter = attrs.find(name);
        if (iter == attrs.end()) { return Status::OK(); }
        const auto value = ToLower(iter->second);
        if (value == "1" || value == "0" || value == "true" || value == "false") {
            return Status::OK();
        }
        UC_ERROR("Validate SQE attr failed name={} value={} reason=invalid_bool", name,
                 iter->second);
        return Status::Error(StatusCode::INVALID_ARGUMENT, name + " is not a valid bool");
    };

    auto status = validateInteger("kv_ns_id", std::numeric_limits<std::uint32_t>::max());
    if (!status.ok()) { return status; }
    status = validateInteger("timeout", std::numeric_limits<std::uint32_t>::max());
    if (!status.ok()) { return status; }
    status = validateInteger("dtype", std::numeric_limits<std::uint8_t>::max());
    if (!status.ok()) { return status; }
    status = validateInteger("dspec", std::numeric_limits<std::uint8_t>::max());
    if (!status.ok()) { return status; }
    status = validateBool("sc");
    if (!status.ok()) { return status; }
    status = validateBool("lr");
    if (!status.ok()) { return status; }
    status =
        validateRequiredPositiveInteger("kernel_count", std::numeric_limits<std::uint32_t>::max());
    if (!status.ok()) { return status; }
    status =
        validateRequiredPositiveInteger("quiet_count", std::numeric_limits<std::uint32_t>::max());
    if (!status.ok()) { return status; }
    return Status::OK();
}

Status LoadTransportConfig(const std::string& configPath, TransportConfig& config)
{
    std::ifstream configFile{configPath};
    if (!configFile.is_open()) {
        return Status::Error(StatusCode::NOT_FOUND,
                             "failed to open asu transport config, path=" + configPath);
    }

    config = TransportConfig{};
    std::string line;
    while (std::getline(configFile, line)) {
        line = TrimConfigValue(line);
        if (line.empty() || line[0] == '#') { continue; }

        const auto pos = line.find('=');
        if (pos == std::string::npos) { continue; }

        const auto key = TrimConfigValue(line.substr(0, pos));
        const auto value = TrimConfigValue(line.substr(pos + 1));
        if (key == "asuName" || key == "asu_name") {
            config.asuName = value;
        } else if (key == "asuId" || key == "asu_id") {
            config.asuId = ParseConfigUint64(value);
        } else if (key == "endpoint" || key == "endpoints") {
            config.endpoints.clear();
            for (const auto& endpointValue : SplitConfigValue(value, ';')) {
                config.endpoints.emplace_back(ParseTransportEndpoint(endpointValue));
            }
        } else if (key == "timeoutMs" || key == "timeout_ms") {
            config.timeoutMs = ParseConfigUint64(value);
        } else if (key == "maxErrorCount" || key == "max_error_count") {
            config.maxErrorCount = static_cast<std::uint32_t>(ParseConfigUint64(value));
        } else if (key == "maxInflightTasks" || key == "max_inflight_tasks") {
            config.maxInflightTasks = static_cast<std::uint32_t>(ParseConfigUint64(value));
        } else if (key == "maxInflightBytes" || key == "max_inflight_bytes") {
            config.maxInflightBytes = ParseConfigUint64(value);
        } else if (ApplyTransportCompletionConfigField(config, key, value)) {
            continue;
        } else if (ApplyTransportBufferConfigField(config, key, value)) {
            continue;
        } else if (ApplyTransportIoNumConfigField(config, key, value)) {
            continue;
        } else if (ApplyTransportProviderConfigField(config, key, value)) {
            continue;
        } else if (ApplyTransportDeviceConfigField(config, key, value)) {
            continue;
        } else {
            config.attrs[key] = value;
        }
    }
    return ValidateSqeRequestAttrs(config.attrs);
}

}  // namespace kv
