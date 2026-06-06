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
#include "transport_config_parser.h"
#include <fstream>
#include <utility>
#include "asu_transport/asu_transport.h"
#include "config_parser_common.h"

namespace UC::ASU {
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
        } else if (key == "queryTimeoutMs" || key == "query_timeout_ms") {
            config.queryTimeoutMs = ParseConfigUint64(value);
        } else if (key == "loadTimeoutMs" || key == "load_timeout_ms") {
            config.loadTimeoutMs = ParseConfigUint64(value);
        } else if (key == "storeTimeoutMs" || key == "store_timeout_ms") {
            config.storeTimeoutMs = ParseConfigUint64(value);
        } else if (key == "maxInflightTasks" || key == "max_inflight_tasks") {
            config.maxInflightTasks = static_cast<std::uint32_t>(ParseConfigUint64(value));
        } else if (key == "maxInflightBytes" || key == "max_inflight_bytes") {
            config.maxInflightBytes = ParseConfigUint64(value);
        } else if (ApplyTransportBufferConfigField(config, key, value)) {
            continue;
        } else if (ApplyTransportIoNumConfigField(config, key, value)) {
            continue;
        } else {
            config.attrs[key] = value;
        }
    }
    return Status::OK();
}

}  // namespace UC::ASU
