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
#include "view_server.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <utility>
#include "asu_client/asu_client.h"

namespace UC::ASU {
namespace {

std::string Trim(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) { return ""; }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> Split(const std::string& value, char delimiter)
{
    std::vector<std::string> parts;
    std::stringstream stream{value};
    std::string part;
    while (std::getline(stream, part, delimiter)) {
        part = Trim(part);
        if (!part.empty()) { parts.emplace_back(std::move(part)); }
    }
    return parts;
}

std::uint64_t ParseUint64(const std::string& value) { return std::stoull(value, nullptr, 0); }

Protocol ToTransportProtocol(const std::string& value)
{
    auto protocol = value;
    std::transform(protocol.begin(), protocol.end(), protocol.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    if (protocol == "UB" || protocol == "UBOE") { return Protocol::UB; }
    if (protocol == "ROCE") { return Protocol::ROCE; }
    if (protocol == "TCP") { return Protocol::TCP; }
    return Protocol::TCP;
}

AsuInfo ExtractAsuInfo(const TransportConfig& config)
{
    AsuInfo info;
    info.endpoints = config.endpoints;
    return info;
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

void SetEndpointAttr(AsuEndpoint& endpoint, const std::string& key, const std::string& value)
{
    endpoint.attrs[key] = value;
}

AsuEndpoint ParseAsuEndpoint(const std::string& value)
{
    AsuEndpoint endpoint;
    if (value.find('=') == std::string::npos) {
        auto parts = Split(value, ':');
        if (!parts.empty()) { endpoint.ip = parts[0]; }
        if (parts.size() > 1) { endpoint.port = static_cast<std::uint16_t>(ParseUint64(parts[1])); }
        if (parts.size() > 2) {
            endpoint.protocol = ToTransportProtocol(parts[2]);
            SetEndpointAttr(endpoint, "protocol", parts[2]);
        }
        return endpoint;
    }

    for (const auto& item : Split(value, ',')) {
        const auto pos = item.find('=');
        if (pos == std::string::npos) { continue; }

        const auto key = Trim(item.substr(0, pos));
        const auto fieldValue = Trim(item.substr(pos + 1));
        if (key == "protocol") {
            endpoint.protocol = ToTransportProtocol(fieldValue);
            SetEndpointAttr(endpoint, "protocol", fieldValue);
        } else if (key == "placement") {
            SetEndpointAttr(endpoint, "placement", fieldValue);
        } else if (key == "port") {
            endpoint.port = static_cast<std::uint16_t>(ParseUint64(fieldValue));
        } else if (key == "local.comm_id" || key == "localCommId") {
            endpoint.ip = fieldValue;
        } else if (key == "local.phy_device_id" || key == "localPhyDeviceId") {
            endpoint.deviceId = static_cast<std::int32_t>(ParseUint64(fieldValue));
        } else if (key == "tc") {
            SetEndpointAttr(endpoint, "tc", fieldValue);
        } else if (key == "sl") {
            SetEndpointAttr(endpoint, "sl", fieldValue);
        } else if (key == "send_size" || key == "sendSize") {
            SetEndpointAttr(endpoint, "send_size", fieldValue);
        } else if (key == "flag_size" || key == "flagSize") {
            SetEndpointAttr(endpoint, "flag_size", fieldValue);
        } else if (key == "remote_send_addr" || key == "remoteSendAddr") {
            SetEndpointAttr(endpoint, "remote_send_addr", fieldValue);
        } else if (key == "remote_flag_addr" || key == "remoteFlagAddr") {
            SetEndpointAttr(endpoint, "remote_flag_addr", fieldValue);
        }
    }
    return endpoint;
}

AsuInfo ParseAsuInfo(const std::string& value)
{
    AsuInfo info;
    for (const auto& endpointValue : Split(value, ';')) {
        info.endpoints.emplace_back(ParseAsuEndpoint(endpointValue));
    }
    return info;
}

bool HasKnownViewEpoch(const GlobalView& view) { return view.viewEpoch != 0; }

class ConfigFileViewServer final : public ViewServer {
public:
    explicit ConfigFileViewServer(std::string configPath) : configPath_(std::move(configPath)) {}

    Status GetGlobalView(GlobalView& view) override
    {
        std::ifstream configFile{configPath_};
        if (!configFile.is_open()) {
            return Status::Error(StatusCode::NOT_FOUND,
                                 "failed to open global view config, path=" + configPath_);
        }

        GlobalView nextView;
        std::string line;
        while (std::getline(configFile, line)) {
            line = Trim(line);
            if (line.empty() || line[0] == '#') { continue; }

            const auto pos = line.find('=');
            if (pos == std::string::npos) { continue; }

            const auto key = Trim(line.substr(0, pos));
            const auto value = Trim(line.substr(pos + 1));
            if (key == "viewEpoch" || key == "view_epoch") {
                nextView.viewEpoch = std::stoull(value);
            } else if (key == "viewId" || key == "view_id") {
                nextView.viewId = std::stoull(value);
            } else if (key == "createTimeMs" || key == "create_time_ms") {
                nextView.createTimeMs = std::stoull(value);
            } else if (key == "expireTimeMs" || key == "expire_time_ms") {
                nextView.expireTimeMs = std::stoull(value);
            } else if (key == "asuIds" || key == "asu_ids") {
                nextView.asuMap.clear();
                for (const auto& asuId : Split(value, ',')) {
                    nextView.asuMap.emplace(std::stoull(asuId), AsuInfo{});
                }
            } else {
                AsuId asuId{0};
                if (TryParseAsuInfoKey(key, asuId)) {
                    nextView.asuMap[asuId] = ParseAsuInfo(value);
                }
            }
        }

        view = std::move(nextView);
        return Status::OK();
    }

private:
    std::string configPath_;
};

class ConfigBackedViewServer final : public ViewServer {
public:
    explicit ConfigBackedViewServer(GlobalView view) : view_(std::move(view)) {}

    Status GetGlobalView(GlobalView& view) override
    {
        view = view_;
        return Status::OK();
    }

private:
    GlobalView view_;
};

}  // namespace

void ApplyAsuInfoToTransportConfig(const AsuInfo& info, TransportConfig& config)
{
    if (info.endpoints.empty()) { return; }

    config.endpoints = info.endpoints;
}

GlobalView BuildConfigGlobalView(const AsuClientConfig& config)
{
    GlobalView view;
    for (const auto& transportConfig : config.transportConfigs) {
        view.asuMap.emplace(transportConfig.asuId, ExtractAsuInfo(transportConfig));
    }
    return view;
}

bool ViewServer::ShouldPublishView(const GlobalView& publishedView,
                                   const GlobalView& fetchedView) const
{
    if (!HasKnownViewEpoch(fetchedView) || !HasKnownViewEpoch(publishedView)) { return true; }
    return fetchedView.viewEpoch > publishedView.viewEpoch;
}

bool ViewServer::ShouldRefreshView(const Status& status) const
{
    switch (status.code) {
        case StatusCode::CONNECTION_ERROR:
        case StatusCode::IO_ERROR:
        case StatusCode::TIMEOUT:
        case StatusCode::NOT_FOUND:
        case StatusCode::BUFFER_NOT_REGISTERED: return true;
        default: return false;
    }
}

bool ViewServer::ShouldRefreshView(const TaskResult& result) const
{
    if (ShouldRefreshView(result.status)) { return true; }
    return std::any_of(result.entryStatus.begin(), result.entryStatus.end(),
                       [this](const Status& status) { return ShouldRefreshView(status); });
}

std::shared_ptr<ViewServer> CreateDefaultViewServer(const AsuClientConfig& config)
{
    if (config.viewServiceAddrs.empty()) {
        return std::make_shared<ConfigBackedViewServer>(BuildConfigGlobalView(config));
    }
    return std::make_shared<ConfigFileViewServer>(config.viewServiceAddrs.front());
}

}  // namespace UC::ASU
