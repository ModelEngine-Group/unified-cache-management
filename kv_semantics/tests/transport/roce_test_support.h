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

// Shared helpers for the CPU-over-RoCE provider tests.  They keep the
// handshake state machine tests free of the socket boilerplate each of them
// would otherwise repeat.
#ifndef KV_SEMANTICS_TESTS_TRANSPORT_ROCE_TEST_SUPPORT_H
#define KV_SEMANTICS_TESTS_TRANSPORT_ROCE_TEST_SUPPORT_H

#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include "kv_types.h"
#include "roce/cpu_kv_over_roce_trans_provider.h"

namespace kv::test {

// Binds an ephemeral port, notes the number and releases it again.  The window
// between the close and the bind inside the test is small enough for a unit test.
inline std::uint16_t ReserveTcpPort()
{
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { return 0; }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (::bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(fd);
        return 0;
    }
    socklen_t length = sizeof(address);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
        ::close(fd);
        return 0;
    }
    ::close(fd);
    return ntohs(address.sin_port);
}

// The unit tests exercise the handshake state machine, not the hardware, so the
// queue pairs are the simulated backend.  The verbs backend needs a machine that
// owns an HCA.
inline TransportConfig MakeClientConfig(std::uint32_t handshakeTimeoutMs = 4000)
{
    TransportConfig config;
    config.providerType = TransProviderType::CPU;
    config.attrs["roce_oob_role"] = "client";
    config.attrs["roce_oob_connect_timeout_ms"] = "2000";
    config.attrs["roce_handshake_timeout_ms"] = std::to_string(handshakeTimeoutMs);
    config.attrs["roce_qp_backend"] = "simulated";
    config.attrs["roce_gid_index"] = "0";
    return config;
}

inline TransportConfig MakeServerConfig()
{
    TransportConfig config;
    config.providerType = TransProviderType::CPU;
    config.attrs["roce_oob_role"] = "server";
    config.attrs["roce_oob_bind_ip"] = "127.0.0.1";
    config.attrs["roce_oob_connect_timeout_ms"] = "2000";
    config.attrs["roce_handshake_timeout_ms"] = "4000";
    config.attrs["roce_qp_backend"] = "simulated";
    config.attrs["roce_gid_index"] = "0";
    return config;
}

// Creates the connection on the client side, retrying while the server thread is
// still on its way into accept().
inline Status ConnectWithRetry(TransProvider& provider, std::uint16_t port, std::uint32_t qpNum,
                               std::vector<TransProvider::ConnectionHandle>& handles)
{
    Status status;
    for (int attempt = 0; attempt < 100; ++attempt) {
        status = provider.CreateConnection("127.0.0.1", "127.0.0.1", port, qpNum, 2000, handles);
        if (status.ok()) { return status; }
        // Only the accept race is worth retrying; a handshake rejection is final.
        if (status.message.find("out-of-band TCP connect") == std::string::npos) { return status; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return status;
}

// Connects a raw channel to the server side, retrying while the listener is on
// its way into accept().
inline Status ConnectChannelWithRetry(std::uint16_t port, roce::TcpChannel& channel)
{
    std::string error;
    Status status;
    for (int attempt = 0; attempt < 100; ++attempt) {
        status = roce::TcpChannel::Connect("127.0.0.1", "127.0.0.1", port, 500, channel, error);
        if (status.ok()) { return status; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return Status::Error(status.code, error);
}

// Sends one already encoded frame over a raw channel.
inline Status SendRawFrame(const roce::TcpChannel& channel, const std::vector<std::uint8_t>& frame)
{
    std::string error;
    Status status = channel.SendAll(frame.data(), frame.size(), 2000, error);
    if (status.ok()) { return status; }
    return Status::Error(status.code, error);
}

}  // namespace kv::test

#endif  // KV_SEMANTICS_TESTS_TRANSPORT_ROCE_TEST_SUPPORT_H
