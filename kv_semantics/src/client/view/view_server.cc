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
#include <fstream>
#include <utility>
#include "kv_client.h"
#include "parser_common.h"
#include "status_utils.h"
#include "utils/config_utils.h"

namespace kv {
namespace {

AsuInfo ExtractAsuInfo(const TransportConfig& config)
{
    AsuInfo info;
    info.endpoints = config.endpoints;
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
            const auto message = "failed to open global view config, path=" + configPath_;
            return ASU_LOG_ERROR_STATUS(StatusCode::NOT_FOUND, message);
        }

        GlobalView nextView;
        std::string line;
        while (std::getline(configFile, line)) {
            line = TrimConfigValue(line);
            if (line.empty() || line[0] == '#') { continue; }

            const auto pos = line.find('=');
            if (pos == std::string::npos) { continue; }

            const auto key = TrimConfigValue(line.substr(0, pos));
            const auto value = TrimConfigValue(line.substr(pos + 1));
            if (key == "viewEpoch" || key == "view_epoch") {
                nextView.viewEpoch = ParseConfigUint64(value);
            } else if (key == "viewId" || key == "view_id") {
                nextView.viewId = ParseConfigUint64(value);
            } else if (key == "createTimeMs" || key == "create_time_ms") {
                nextView.createTimeMs = ParseConfigUint64(value);
            } else if (key == "expireTimeMs" || key == "expire_time_ms") {
                nextView.expireTimeMs = ParseConfigUint64(value);
            } else if (key == "asuIds" || key == "asu_ids") {
                nextView.asuMap.clear();
                for (const auto& asuId : SplitConfigValue(value, ',')) {
                    nextView.asuMap.emplace(ParseConfigUint64(asuId), AsuInfo{});
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
    auto viewConfigPath = config.attrs.find("view.config_path");
    if (viewConfigPath != config.attrs.end() && !viewConfigPath->second.empty()) {
        return std::make_shared<ConfigFileViewServer>(viewConfigPath->second);
    }
    if (config.viewServiceAddrs.empty()) {
        return std::make_shared<ConfigBackedViewServer>(BuildConfigGlobalView(config));
    }
    return std::make_shared<ConfigFileViewServer>(config.viewServiceAddrs.front());
}

}  // namespace kv
