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
#include "config_parser_common.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <utility>

namespace UC::ASU {
namespace {

void SetEndpointAttr(AsuEndpoint& endpoint, const std::string& key, const std::string& value)
{
    endpoint.attrs[key] = value;
}

void ApplyTransportEndpointField(AsuEndpoint& endpoint, const std::string& key,
                                 const std::string& value)
{
    if (key == "ip" || key == "local.comm_id" || key == "localCommId") {
        endpoint.ip = value;
    } else if (key == "port") {
        endpoint.port = static_cast<std::uint16_t>(ParseConfigUint64(value));
    } else if (key == "protocol") {
        endpoint.protocol = ParseConfigProtocol(value);
    } else if (key == "numa_node" || key == "numaNode") {
        endpoint.numaNode = static_cast<std::int32_t>(ParseConfigUint64(value));
    } else if (key == "hca_name" || key == "hcaName") {
        endpoint.hcaName = value;
    } else if (key == "hca_port" || key == "hcaPort") {
        endpoint.hcaPort = static_cast<std::uint8_t>(ParseConfigUint64(value));
    } else {
        endpoint.attrs[key] = value;
    }
}

void ApplyClientViewEndpointField(AsuEndpoint& endpoint, const std::string& key,
                                  const std::string& value)
{
    if (key == "protocol") {
        endpoint.protocol = ParseConfigProtocol(value);
        SetEndpointAttr(endpoint, "protocol", value);
    } else if (key == "placement") {
        SetEndpointAttr(endpoint, "placement", value);
    } else if (key == "port") {
        endpoint.port = static_cast<std::uint16_t>(ParseConfigUint64(value));
    } else if (key == "local.comm_id" || key == "localCommId") {
        endpoint.ip = value;
    } else if (key == "tc") {
        SetEndpointAttr(endpoint, "tc", value);
    } else if (key == "sl") {
        SetEndpointAttr(endpoint, "sl", value);
    } else if (key == "send_size" || key == "sendSize") {
        SetEndpointAttr(endpoint, "send_size", value);
    } else if (key == "flag_size" || key == "flagSize") {
        SetEndpointAttr(endpoint, "flag_size", value);
    } else if (key == "remote_send_addr" || key == "remoteSendAddr") {
        SetEndpointAttr(endpoint, "remote_send_addr", value);
    } else if (key == "remote_flag_addr" || key == "remoteFlagAddr") {
        SetEndpointAttr(endpoint, "remote_flag_addr", value);
    }
}

}  // namespace

std::string TrimConfigValue(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) { return ""; }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> SplitConfigValue(const std::string& value, char delimiter)
{
    std::vector<std::string> parts;
    std::stringstream stream{value};
    std::string part;
    while (std::getline(stream, part, delimiter)) {
        part = TrimConfigValue(part);
        if (!part.empty()) { parts.emplace_back(std::move(part)); }
    }
    return parts;
}

std::uint64_t ParseConfigUint64(const std::string& value) { return std::stoull(value, nullptr, 0); }

Protocol ParseConfigProtocol(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    if (value == "UB" || value == "UBOE") { return Protocol::UB; }
    if (value == "ROCE") { return Protocol::ROCE; }
    if (value == "TCP") { return Protocol::TCP; }
    return Protocol::TCP;
}

TransProviderType ParseConfigTransProviderType(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    if (value == "FAKE") { return TransProviderType::FAKE; }
    if (value == "AIV") { return TransProviderType::AIV; }
    if (value == "AICPU") { return TransProviderType::AICPU; }
    return TransProviderType::UNSUPPORTED;
}

bool ApplyTransportBufferConfigField(TransportConfig& config, const std::string& key,
                                     const std::string& value)
{
    if (key == "sendBufferSlotSize" || key == "send_buffer_slot_size" ||
        key == "ioBuffer.sendBufferSlotSize" || key == "io_buffer.send_buffer_slot_size") {
        config.sendBufferSlotSize = static_cast<std::size_t>(ParseConfigUint64(value));
    } else if (key == "sendBufferSlotNum" || key == "send_buffer_slot_num" ||
               key == "ioBuffer.sendBufferSlotNum" || key == "io_buffer.send_buffer_slot_num") {
        config.sendBufferSlotNum = static_cast<std::size_t>(ParseConfigUint64(value));
    } else if (key == "flagBufferSlotSize" || key == "flag_buffer_slot_size" ||
               key == "ioBuffer.flagBufferSlotSize" || key == "io_buffer.flag_buffer_slot_size") {
        config.flagBufferSlotSize = static_cast<std::size_t>(ParseConfigUint64(value));
    } else if (key == "flagBufferSlotNum" || key == "flag_buffer_slot_num" ||
               key == "ioBuffer.flagBufferSlotNum" || key == "io_buffer.flag_buffer_slot_num") {
        config.flagBufferSlotNum = static_cast<std::size_t>(ParseConfigUint64(value));
    } else {
        return false;
    }
    return true;
}

bool ApplyTransportIoNumConfigField(TransportConfig& config, const std::string& key,
                                    const std::string& value)
{
    if (key == "batchLoadIoNum" || key == "batch_load_io_num") {
        config.asuBatchLoadIoNum = static_cast<std::size_t>(ParseConfigUint64(value));
    } else if (key == "batchStoreIoNum" || key == "batch_store_io_num") {
        config.asuBatchStoreIoNum = static_cast<std::size_t>(ParseConfigUint64(value));
    } else if (key == "deleteIoNum" || key == "delete_io_num") {
        config.asuDeleteIoNum = static_cast<std::size_t>(ParseConfigUint64(value));
    } else if (key == "queryIoNum" || key == "query_io_num") {
        config.asuQueryIoNum = static_cast<std::size_t>(ParseConfigUint64(value));
    } else {
        return false;
    }
    return true;
}

bool ApplyTransportCompletionConfigField(TransportConfig& config, const std::string& key,
                                         const std::string& value)
{
    if (key != "completionPollSpinLimit" && key != "completion_poll_spin_limit") { return false; }
    config.completionPollSpinLimit = static_cast<std::size_t>(ParseConfigUint64(value));
    return true;
}

bool ApplyTransportProviderConfigField(TransportConfig& config, const std::string& key,
                                       const std::string& value)
{
    if (key == "providerBackend" || key == "provider_backend" || key == "transProviderBackend" ||
        key == "trans_provider_backend" || key == "providerType" || key == "provider_type" ||
        key == "transProviderType" || key == "trans_provider_type") {
        config.providerType = ParseConfigTransProviderType(value);
    } else {
        return false;
    }
    return true;
}

bool ApplyTransportDeviceConfigField(TransportConfig& config, const std::string& key,
                                     const std::string& value)
{
    if (key == "deviceId" || key == "device_id") {
        config.deviceId = static_cast<std::int32_t>(ParseConfigUint64(value));
        return true;
    }
    return false;
}

bool TryParseAsuInfoKey(const std::string& key, AsuId& asuId)
{
    constexpr const char* kCamelPrefix = "asuInfo.";
    constexpr const char* kSnakePrefix = "asu_info.";
    if (key.rfind(kCamelPrefix, 0) == 0) {
        asuId = std::stoull(key.substr(std::string{kCamelPrefix}.size()));
        return true;
    }
    if (key.rfind(kSnakePrefix, 0) == 0) {
        asuId = std::stoull(key.substr(std::string{kSnakePrefix}.size()));
        return true;
    }
    return false;
}

bool TryGetTransportAttrKey(const std::string& key, std::string& attrKey)
{
    constexpr const char* kCamelPrefix = "transport.";
    if (key.rfind(kCamelPrefix, 0) == 0) {
        attrKey = key.substr(std::string{kCamelPrefix}.size());
        return !attrKey.empty();
    }
    return false;
}

AsuEndpoint ParseTransportEndpoint(const std::string& value)
{
    AsuEndpoint endpoint;
    if (value.find('=') == std::string::npos) {
        auto parts = SplitConfigValue(value, ':');
        if (!parts.empty()) { endpoint.ip = parts[0]; }
        if (parts.size() > 1) {
            endpoint.port = static_cast<std::uint16_t>(ParseConfigUint64(parts[1]));
        }
        if (parts.size() > 2) { endpoint.protocol = ParseConfigProtocol(parts[2]); }
        return endpoint;
    }

    for (const auto& item : SplitConfigValue(value, ',')) {
        const auto pos = item.find('=');
        if (pos == std::string::npos) { continue; }
        ApplyTransportEndpointField(endpoint, TrimConfigValue(item.substr(0, pos)),
                                    TrimConfigValue(item.substr(pos + 1)));
    }
    return endpoint;
}

AsuEndpoint ParseClientViewEndpoint(const std::string& value)
{
    AsuEndpoint endpoint;
    if (value.find('=') == std::string::npos) {
        auto parts = SplitConfigValue(value, ':');
        if (!parts.empty()) { endpoint.ip = parts[0]; }
        if (parts.size() > 1) {
            endpoint.port = static_cast<std::uint16_t>(ParseConfigUint64(parts[1]));
        }
        if (parts.size() > 2) {
            endpoint.protocol = ParseConfigProtocol(parts[2]);
            SetEndpointAttr(endpoint, "protocol", parts[2]);
        }
        return endpoint;
    }

    for (const auto& item : SplitConfigValue(value, ',')) {
        const auto pos = item.find('=');
        if (pos == std::string::npos) { continue; }
        ApplyClientViewEndpointField(endpoint, TrimConfigValue(item.substr(0, pos)),
                                     TrimConfigValue(item.substr(pos + 1)));
    }
    return endpoint;
}

}  // namespace UC::ASU
