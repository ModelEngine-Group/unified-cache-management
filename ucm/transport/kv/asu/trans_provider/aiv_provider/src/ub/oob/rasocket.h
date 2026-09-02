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

#pragma once

#include <cstdint>
#include <string>
#include "src/ub/neg_cap.h"
#include "src/ub/oob/tcp_client.h"
#include "src/ub/oob/transport.h"
#include "src/ub/status.h"

namespace umc::comm::oob {

class RaSocketOobTransport : public OobTransport {
public:
    struct Config {
        uint32_t phyId{0};
        std::string localIp;
        std::string host{"127.0.0.1"};
        uint16_t port{0};
        uint32_t connectTimeoutMs{5000};
        uint32_t ioTimeoutMs{5000};
        TcpClientLocalEndpoint local;
        uint16_t kato{5000};
        uint16_t cmdDepth{1024};
    };

    explicit RaSocketOobTransport(Config cfg);
    ~RaSocketOobTransport() override;

    RaSocketOobTransport(const RaSocketOobTransport&) = delete;
    RaSocketOobTransport& operator=(const RaSocketOobTransport&) = delete;

    static bool Available();

    static bool QueryLocalNicIp(uint32_t phyId, std::string* outIp);

    UbStatus Negotiate(RemoteEndpointSet* outRemotes) override;
    const char* Name() const override { return "RaSocket"; }

    bool HasNegCaps() const { return hasNegCaps_; }
    const ub_client_cap& LocalCap() const { return localCap_; }
    const ub_server_cap& RemoteCap() const { return remoteCap_; }

    UbStatus CheckKvLimits(uint32_t keyLen, uint64_t valueBytes) const;

private:
    UbStatus NegotiateV2(void* fdHandle, RemoteEndpointSet* out);

    Config cfg_;
    bool hasNegCaps_{false};
    ub_client_cap localCap_{};
    ub_server_cap remoteCap_{};
};

}  // namespace umc::comm::oob
