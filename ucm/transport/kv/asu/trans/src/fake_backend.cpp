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
#include "asu_transport/fake_backend.h"
#include <string>
#include <utility>

namespace UC::ASU {

void PatchFakeBackendTransportConfig(TransportConfig& config, const FakeBackendConfig& fakeConfig)
{
    config.providerType = TransProviderType::FAKE;
    config.attrs.try_emplace("kernel_count", "1");
    config.attrs.try_emplace("quiet_count", "1");
    config.attrs["kv_ns_id"] = std::to_string(config.asuId);
    config.attrs.try_emplace("dtype", "0");
    config.attrs.try_emplace("dspec", "0");
    config.attrs.try_emplace("lr", "false");
    config.attrs["sc"] = "true";
    config.attrs["fake_backend.path"] = fakeConfig.storePath;
    config.attrs["fake_backend.latency_ms"] = std::to_string(fakeConfig.latencyMs);
    config.attrs["fake_backend.device_id"] = std::to_string(fakeConfig.deviceId);
    if (config.endpoints.empty()) {
        AsuEndpoint endpoint;
        endpoint.ip = "fake_backend";
        endpoint.port = 19001;
        endpoint.protocol = Protocol::TCP;
        endpoint.deviceId = fakeConfig.deviceId;
        config.endpoints.emplace_back(std::move(endpoint));
    }
}

}  // namespace UC::ASU
