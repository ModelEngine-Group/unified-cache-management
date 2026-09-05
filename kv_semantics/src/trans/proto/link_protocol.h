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
#include <cstring>
#include <string>
#include <vector>
#include "types.h"

namespace kv {

constexpr std::uint32_t kMsgHeaderSize = 16;
constexpr std::uint32_t kMsgVersion = 1;

// LinkProtocolCmd defines the message types for the KV connection protocol.
enum class LinkProtocolCmd : std::uint8_t {
    Negotiate = 0,
    Handshake = 1,
    HandshakeDone = 3,
    Disconnect = 4
};

constexpr std::size_t kNegotiatePayloadSize = 160;  // 32 + 128
constexpr std::size_t kHandshakePayloadSize = 160;
constexpr std::size_t kDisconnectPayloadSize = 8;

class MsgHeader {
public:
    std::uint32_t crc{0};
    std::uint8_t ver{kMsgVersion};
    LinkProtocolCmd cmd{LinkProtocolCmd::Negotiate};
    std::uint8_t pad[6]{0};
    std::uint32_t len{0};

    void Pack(std::vector<std::uint8_t>& buffer) const
    {
        std::memcpy(&buffer[0], &crc, 4);
        buffer[4] = ver;
        buffer[5] = static_cast<std::uint8_t>(cmd);
        std::memcpy(&buffer[12], &len, 4);
    }

    static MsgHeader Unpack(const std::uint8_t* data)
    {
        MsgHeader h;
        std::memcpy(&h.crc, &data[0], 4);
        h.ver = data[4];
        h.cmd = static_cast<LinkProtocolCmd>(data[5]);
        std::memcpy(&h.len, &data[12], 4);
        return h;
    }
};

class NegotiateRequest {
public:
    std::uint32_t cap{0};
    std::uint32_t private_len{4};
    std::uint8_t major_version{1};
    std::uint8_t minor_version{0};
    std::uint16_t kato{0};
};

class HandshakeRequest {
public:
    std::uint8_t gid[16]{0};
    std::uint16_t lid{0};
    std::uint8_t mtu{0};
    std::uint8_t total_qp_num{0};
    std::uint8_t sl{0};
    std::uint8_t traffic_class{0};
    std::uint8_t rnr_timer{0};
    std::uint8_t rnr_retry_cnt{0};
    std::uint8_t timeout{0};
    std::uint8_t retry_cnt{0};
    std::uint8_t qp_rd_atom{0};
    std::uint8_t rsv{0};
    std::uint32_t start_psn{0};
    std::uint32_t qpn[32]{0};
};

class DisconnectRequest {
public:
    std::uint32_t local_qpn{0};
    std::uint32_t remote_qpn{0};
};

class NegotiateSqe {
public:
    NegotiateSqe() = default;

    const void* Data() const { return buffer.data(); }
    std::size_t Size() const { return buffer.size(); }

    Status Pack(const NegotiateRequest& req);
    Status Validate() const;

private:
    std::vector<std::uint8_t> buffer;
};

class HandshakeSqe {
public:
    HandshakeSqe() = default;

    const void* Data() const { return buffer.data(); }
    std::size_t Size() const { return buffer.size(); }

    Status Pack(const HandshakeRequest& req);
    Status Validate() const;

private:
    std::vector<std::uint8_t> buffer;
};

class HandshakeDoneSqe {
public:
    HandshakeDoneSqe() = default;

    const void* Data() const { return buffer.data(); }
    std::size_t Size() const { return buffer.size(); }

    Status Pack();

private:
    std::vector<std::uint8_t> buffer;
};

class DisconnectSqe {
public:
    DisconnectSqe() = default;

    const void* Data() const { return buffer.data(); }
    std::size_t Size() const { return buffer.size(); }

    Status Pack(const DisconnectRequest& req);

private:
    std::vector<std::uint8_t> buffer;
};

}  // namespace kv
