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

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "src/ub/status.h"

namespace umc::comm::oob {

enum class NetAddrKind : uint8_t {
    None = 0,
    Uboe4 = 1,  // UBoE + IPv4
    Uboe6 = 2,  // UBoE + IPv6
};

struct RemoteNetAddr {
    NetAddrKind kind{NetAddrKind::None};
    uint8_t prefixLen{0};
    uint16_t family{0};  // AF_INET / AF_INET6 / 0
    uint64_t vlan{0};
    std::array<uint8_t, 6> mac{};
    std::array<uint8_t, 16> ipRaw{};

    bool IsUboe() const { return kind != NetAddrKind::None; }
};

struct RemoteJettyDescriptor {
    std::string peerName;
    std::array<uint8_t, 16> hccpEidRaw{};
    uint32_t jettyId{0};

    std::array<uint8_t, 64> qpKeyRaw{};
    uint8_t qpKeySize{0};

    uint32_t segId{0};
    std::array<uint8_t, 128> memKeyRaw{};
    uint8_t memKeySize{0};

    uint32_t tokenId{0};
    uint32_t tokenValue{0};
    uint64_t remoteAddr{0};
    uint64_t remoteSize{0};

    std::string usage;
    uint32_t signalSlotCount{0};
    uint32_t signalSlotStride{0};
    uint64_t signalSlotBase{0};
    uint64_t dataBase{0};

    RemoteNetAddr netAddr{};

    uint64_t peerTpHandle{0};
    uint64_t tag{0};
    uint32_t peerTxPsn{0};

    uint32_t uasid{0};
    uint8_t connMode{0};  // 0=Rc / 1=Rm
};

struct RemoteEndpointSet {
    int schemaVersion{0};
    std::string clusterName;
    std::string protocol;                        // "UB_PROTO"
    std::string tokenPolicy;                     // "PLAIN_TEXT" / "SIGNED" / "ENCRYPTED"
    std::vector<RemoteJettyDescriptor> jetties;  // flattened across peers
};

class OobTransport {
public:
    virtual ~OobTransport() = default;
    virtual UbStatus Negotiate(RemoteEndpointSet* outRemotes) = 0;
    virtual const char* Name() const = 0;
};

}  // namespace umc::comm::oob
