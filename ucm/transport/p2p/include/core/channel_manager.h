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

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "common/thread_pool.h"
#include "control/control_protocol.h"
#include "protocols/tcp/tcp_transport.h"

namespace transport {

class Channel final {
public:
    explicit Channel(TcpTransport::Socket socket, bool outgoing = false);
    ~Channel();

    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

private:
    friend class ChannelManager;

    TcpTransport::Socket socket_;
    bool outgoing_ = false;
    ManagerID peer_;
    Metadata receive_buffer_;
    std::mutex send_mutex_;
    std::unordered_set<TransportProtocol> connected_protocols_;
};

class ChannelManager final {
public:
    using Handler = std::function<Status(ManagerMessageType, TransportProtocol, const ManagerID&,
                                         const Metadata&, Metadata&)>;

    ChannelManager();
    ~ChannelManager();

    Status Init(const ManagerID& local, const Endpoint& endpoint, Handler handler);
    Status Request(const Endpoint& peer, ManagerMessageType type, TransportProtocol protocol,
                   const Metadata& payload, Metadata& response);
    Status Send(const Endpoint& peer, const Metadata& payload);
    Status Receive(ManagerID& peer, Metadata& payload);
    Status Close(const Endpoint& peer);
    bool IsConnected(const Endpoint& peer, TransportProtocol protocol) const;
    void SetConnected(const Endpoint& peer, TransportProtocol protocol, bool connected);
    std::vector<TransportProtocol> ConnectedProtocols(const Endpoint& peer) const;
    std::vector<ManagerID> Peers() const;
    void Shutdown();

private:
    struct Message {
        ManagerMessageType type = ManagerMessageType::Data;
        uint64_t request_id = 0;
        TransportProtocol protocol = TransportProtocol::Hixl;
        ManagerID source;
        Status status = Status::OK();
        Metadata payload;
    };

    struct PendingRequest {
        std::weak_ptr<Channel> channel;
        std::mutex mutex;
        std::condition_variable ready;
        bool completed = false;
        Status status = Status::Error();
        Metadata payload;
    };

    struct ReceivedMessage {
        ManagerID peer;
        Metadata payload;
    };

    struct PeerTaskQueue {
        std::deque<ThreadPool::Task> tasks;
    };

    void AddAccepted(TcpTransport::Socket socket);
    Status GetOrConnect(const Endpoint& peer, std::shared_ptr<Channel>& channel);
    static Status EncodeMessage(const Message& message, Metadata& frame);
    static Status DecodeMessage(const Metadata& frame, Message& message);
    Status SendMessage(const std::shared_ptr<Channel>& channel, const Message& message);
    Status SubmitTask(const ManagerID& peer, ThreadPool::Task task);
    void RunTasks(const ManagerID& peer, const std::shared_ptr<PeerTaskQueue>& queue);
    void ReceiveLoop();
    void ProcessFrames(const std::shared_ptr<Channel>& channel);
    void HandleMessage(const std::shared_ptr<Channel>& channel, Message message);
    void RemoveChannel(const std::shared_ptr<Channel>& channel, bool cleanup_transports);

    ManagerID local_;
    TcpTransport tcp_;
    std::unique_ptr<ThreadPool> thread_pool_;
    Handler handler_;
    mutable std::mutex mutex_;
    std::unordered_map<TcpTransport::Socket, std::shared_ptr<Channel>> sockets_;
    std::unordered_map<ManagerID, std::shared_ptr<Channel>> peers_;
    std::unordered_map<uint64_t, std::shared_ptr<PendingRequest>> pending_;
    std::unordered_map<ManagerID, std::shared_ptr<PeerTaskQueue>> task_queues_;
    std::deque<ReceivedMessage> received_;
    std::condition_variable received_ready_;
    std::thread receive_thread_;
    int epoll_fd_ = -1;
    bool stopping_ = false;
    uint64_t next_request_id_ = 1;
};

}  // namespace transport
