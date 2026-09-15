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
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <thread>
#include "core/transport.h"

namespace transport {

class TcpTransport final {
public:
    using Socket = int;
    using AcceptHandler = std::function<void(Socket)>;
    static constexpr Socket kInvalidSocket = -1;

    TcpTransport() = default;
    ~TcpTransport();

    TcpTransport(const TcpTransport&) = delete;
    TcpTransport& operator=(const TcpTransport&) = delete;

    Status StartAccepting(const Endpoint& endpoint, AcceptHandler handler);
    Status Connect(const Endpoint& endpoint, Socket& socket) const;
    Status Send(Socket socket, const void* data, std::size_t length) const;
    Status Receive(Socket socket, void* data, std::size_t capacity, std::size_t& received) const;
    void Stop();

    static void Close(Socket socket);

private:
    void AcceptLoop();

    Socket listen_socket_ = kInvalidSocket;
    AcceptHandler accept_handler_;
    std::thread accept_thread_;
    std::atomic<bool> stopping_{false};
};

}  // namespace transport
