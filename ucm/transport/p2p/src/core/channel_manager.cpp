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

#include "core/channel_manager.h"
#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <future>
#include <sys/epoll.h>
#include <unistd.h>
#include <vector>
#include "common/status_utils.h"

namespace transport {
namespace {

constexpr uint32_t kMaxFrameSize = 4 * 1024 * 1024;
constexpr std::size_t kManagerMessageBodySize = sizeof(ManagerMessageProtocol) - sizeof(uint32_t);
constexpr auto kRequestTimeout = std::chrono::seconds(30);
constexpr int kMaxReadyChannels = 64;

bool IsControlRequest(ManagerMessageType type)
{
    return type == ManagerMessageType::ConnectRequest ||
           type == ManagerMessageType::DisconnectRequest;
}

}  // namespace

Channel::Channel(TcpTransport::Socket socket, bool outgoing) : socket_(socket), outgoing_(outgoing)
{
}

Channel::~Channel() { TcpTransport::Close(socket_); }

ChannelManager::ChannelManager() = default;

ChannelManager::~ChannelManager() { Shutdown(); }

Status ChannelManager::EncodeMessage(const Message& message, Metadata& frame)
{
    if (static_cast<uint8_t>(message.type) > static_cast<uint8_t>(ManagerMessageType::Data) ||
        message.source.size() > kMaxFrameSize - kManagerMessageBodySize ||
        message.payload.size() > kMaxFrameSize - kManagerMessageBodySize - message.source.size()) {
        return Status::InvalidParam();
    }
    const auto body_size = kManagerMessageBodySize + message.source.size() + message.payload.size();
    frame.resize(sizeof(uint32_t) + body_size);
    auto* protocol = reinterpret_cast<ManagerMessageProtocol*>(frame.data());
    *protocol = {};
    protocol->body_size = htonl(static_cast<uint32_t>(body_size));
    protocol->request_id = message.request_id;
    protocol->protocol = message.protocol;
    protocol->status = message.status.Underlying();
    protocol->source_size = static_cast<uint32_t>(message.source.size());
    protocol->payload_size = static_cast<uint32_t>(message.payload.size());
    protocol->type = message.type;
    auto* output = frame.data() + sizeof(*protocol);
    if (!message.source.empty()) {
        std::memcpy(output, message.source.data(), message.source.size());
    }
    output += message.source.size();
    if (!message.payload.empty()) {
        std::memcpy(output, message.payload.data(), message.payload.size());
    }
    return Status::OK();
}

Status ChannelManager::DecodeMessage(const Metadata& frame, Message& decoded)
{
    if (frame.size() < sizeof(ManagerMessageProtocol)) { return Status::InvalidParam(); }
    const auto* protocol = reinterpret_cast<const ManagerMessageProtocol*>(frame.data());
    const auto body_size = ntohl(protocol->body_size);
    if (body_size != frame.size() - sizeof(uint32_t) ||
        static_cast<uint8_t>(protocol->type) > static_cast<uint8_t>(ManagerMessageType::Data)) {
        return Status::InvalidParam();
    }
    const auto content_size =
        static_cast<std::size_t>(protocol->source_size) + protocol->payload_size;
    if (content_size != frame.size() - sizeof(*protocol)) { return Status::InvalidParam(); }
    decoded.type = protocol->type;
    decoded.request_id = protocol->request_id;
    decoded.protocol = protocol->protocol;
    decoded.status = Status{protocol->status, {}};
    const auto* content = frame.data() + sizeof(*protocol);
    decoded.source.assign(reinterpret_cast<const char*>(content), protocol->source_size);
    decoded.payload.assign(content + protocol->source_size, content + content_size);
    return Status::OK();
}

Status ChannelManager::Init(const ManagerID& local, const Endpoint& endpoint, Handler handler)
{
    if (local.empty() || !handler) { return Status::InvalidParam(); }
    Shutdown();
    local_ = local;
    thread_pool_ = std::make_unique<ThreadPool>();
    handler_ = std::move(handler);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = false;
    }
    epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
    P2P_RETURN_IF_TRUE(epoll_fd_ < 0, Status::Error(),
                       "transport manager channel epoll creation failed");
    auto rollback = MakeScopeGuard([this]() {
        close(epoll_fd_);
        epoll_fd_ = -1;
    });
    P2P_RETURN_IF_ERROR(tcp_.StartAccepting(endpoint, [this](auto socket) { AddAccepted(socket); }),
                        "transport manager channel listen failed endpoint={}", endpoint.ToString());
    receive_thread_ = std::thread(&ChannelManager::ReceiveLoop, this);
    rollback.Dismiss();
    return Status::OK();
}

Status ChannelManager::SubmitTask(const ManagerID& peer, ThreadPool::Task task)
{
    std::shared_ptr<PeerTaskQueue> queue;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = task_queues_.find(peer);
        if (found != task_queues_.end()) {
            found->second->tasks.push_back(std::move(task));
            return Status::OK();
        }
        queue = std::make_shared<PeerTaskQueue>();
        queue->tasks.push_back(std::move(task));
        task_queues_.emplace(peer, queue);
    }
    const auto status = thread_pool_->Submit([this, peer, queue]() { RunTasks(peer, queue); });
    if (status.Failure()) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = task_queues_.find(peer);
        if (found != task_queues_.end() && found->second == queue) { task_queues_.erase(found); }
    }
    return status;
}

void ChannelManager::RunTasks(const ManagerID& peer, const std::shared_ptr<PeerTaskQueue>& queue)
{
    for (;;) {
        ThreadPool::Task task;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue->tasks.empty()) {
                const auto found = task_queues_.find(peer);
                if (found != task_queues_.end() && found->second == queue) {
                    task_queues_.erase(found);
                }
                return;
            }
            task = std::move(queue->tasks.front());
            queue->tasks.pop_front();
        }
        try {
            task();
        } catch (...) {
        }
    }
}

void ChannelManager::AddAccepted(TcpTransport::Socket socket)
{
    auto channel = std::make_shared<Channel>(socket);
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) { return; }
    epoll_event event{};
    event.events = EPOLLIN | EPOLLRDHUP;
    event.data.fd = socket;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, socket, &event) != 0) {
        UC_ERROR("transport manager channel epoll add failed socket={} errno={}", socket, errno);
        return;
    }
    sockets_[socket] = std::move(channel);
}

Status ChannelManager::GetOrConnect(const Endpoint& endpoint, std::shared_ptr<Channel>& channel)
{
    const auto peer = endpoint.ToString();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = peers_.find(peer);
        if (found != peers_.end()) {
            channel = found->second;
            return Status::OK();
        }
    }
    TcpTransport::Socket socket = TcpTransport::kInvalidSocket;
    P2P_RETURN_IF_ERROR(tcp_.Connect(endpoint, socket),
                        "transport manager channel connect failed peer={}", peer);
    channel = std::make_shared<Channel>(socket, true);
    channel->peer_ = peer;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) { return Status::Error(); }
        epoll_event event{};
        event.events = EPOLLIN | EPOLLRDHUP;
        event.data.fd = socket;
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, socket, &event) != 0) { return Status::Error(); }
        sockets_[socket] = channel;
        peers_[peer] = channel;
    }
    return Status::OK();
}

Status ChannelManager::SendMessage(const std::shared_ptr<Channel>& channel, const Message& message)
{
    Metadata frame;
    P2P_RETURN_IF_ERROR(EncodeMessage(message, frame),
                        "transport manager channel encode failed peer={}", channel->peer_);
    std::lock_guard<std::mutex> lock(channel->send_mutex_);
    return tcp_.Send(channel->socket_, frame.data(), frame.size());
}

Status ChannelManager::Request(const Endpoint& endpoint, ManagerMessageType type,
                               TransportProtocol protocol, const Metadata& payload,
                               Metadata& response)
{
    const auto peer = endpoint.ToString();
    std::shared_ptr<Channel> channel;
    P2P_RETURN_IF_ERROR(GetOrConnect(endpoint, channel),
                        "transport manager channel request failed to acquire peer channel peer={}",
                        peer);
    auto pending = std::make_shared<PendingRequest>();
    pending->channel = channel;
    uint64_t request_id = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        request_id = next_request_id_++;
        pending_[request_id] = pending;
    }
    const auto status =
        SendMessage(channel, Message{type, request_id, protocol, local_, Status::OK(), payload});
    if (status.Failure()) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.erase(request_id);
        return status;
    }
    std::unique_lock<std::mutex> lock(pending->mutex);
    if (!pending->ready.wait_for(lock, kRequestTimeout,
                                 [&pending] { return pending->completed; })) {
        lock.unlock();
        std::lock_guard<std::mutex> manager_lock(mutex_);
        pending_.erase(request_id);
        return Status::Error();
    }
    response = std::move(pending->payload);
    return pending->status;
}

Status ChannelManager::Send(const Endpoint& endpoint, const Metadata& payload)
{
    const auto peer = endpoint.ToString();
    std::shared_ptr<Channel> channel;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = peers_.find(peer);
        if (found == peers_.end() || found->second->connected_protocols_.empty()) {
            return Status::Error();
        }
        channel = found->second;
    }
    Message message;
    message.type = ManagerMessageType::Data;
    message.source = local_;
    message.payload = payload;
    return SendMessage(channel, message);
}

Status ChannelManager::Receive(ManagerID& peer, Metadata& payload)
{
    std::unique_lock<std::mutex> lock(mutex_);
    received_ready_.wait(lock, [this] { return stopping_ || !received_.empty(); });
    if (received_.empty()) { return Status::Error(); }
    auto message = std::move(received_.front());
    received_.pop_front();
    peer = std::move(message.peer);
    payload = std::move(message.payload);
    return Status::OK();
}

void ChannelManager::ReceiveLoop()
{
    std::array<epoll_event, kMaxReadyChannels> events{};
    for (;;) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) { return; }
        }
        const auto ready = epoll_wait(epoll_fd_, events.data(), events.size(), 100);
        if (ready <= 0) { continue; }
        for (int i = 0; i < ready; ++i) {
            std::shared_ptr<Channel> channel;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                const auto found = sockets_.find(events[i].data.fd);
                if (found == sockets_.end()) { continue; }
                channel = found->second;
            }
            if ((events[i].events & EPOLLIN) != 0) { ProcessFrames(channel); }
            if ((events[i].events & (EPOLLERR | EPOLLHUP | EPOLLRDHUP)) != 0) {
                RemoveChannel(channel, true);
            }
        }
    }
}

void ChannelManager::ProcessFrames(const std::shared_ptr<Channel>& channel)
{
    uint8_t chunk[64 * 1024];
    for (;;) {
        std::size_t received = 0;
        const auto status = tcp_.Receive(channel->socket_, chunk, sizeof(chunk), received);
        if (status.Success()) {
            channel->receive_buffer_.insert(channel->receive_buffer_.end(), chunk,
                                            chunk + received);
            continue;
        }
        if (status != Status::Retry()) { RemoveChannel(channel, true); }
        break;
    }
    while (channel->receive_buffer_.size() >= sizeof(uint32_t)) {
        uint32_t encoded_size = 0;
        std::copy_n(channel->receive_buffer_.data(), sizeof(encoded_size),
                    reinterpret_cast<uint8_t*>(&encoded_size));
        const auto size = ntohl(encoded_size);
        if (size > kMaxFrameSize) {
            RemoveChannel(channel, true);
            return;
        }
        if (channel->receive_buffer_.size() < sizeof(uint32_t) + size) { return; }
        Metadata frame(channel->receive_buffer_.begin(),
                       channel->receive_buffer_.begin() + sizeof(uint32_t) + size);
        channel->receive_buffer_.erase(channel->receive_buffer_.begin(),
                                       channel->receive_buffer_.begin() + sizeof(uint32_t) + size);
        Message message;
        if (DecodeMessage(frame, message).Failure()) {
            RemoveChannel(channel, true);
            return;
        }
        if (!IsControlRequest(message.type)) {
            HandleMessage(channel, std::move(message));
            continue;
        }
        const auto peer = message.source;
        if (SubmitTask(peer, [this, channel, message = std::move(message)]() mutable {
                HandleMessage(channel, std::move(message));
            }).Failure()) {
            RemoveChannel(channel, true);
            return;
        }
    }
}

void ChannelManager::HandleMessage(const std::shared_ptr<Channel>& channel, Message message)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (channel->peer_.empty()) {
            channel->peer_ = message.source;
            const auto existing = peers_.find(message.source);
            // Crossed connections keep the TCP channel opened by the smaller ManagerID.
            if (existing == peers_.end() || ((local_ < message.source) == channel->outgoing_)) {
                peers_[message.source] = channel;
            }
        }
    }
    if (message.type == ManagerMessageType::Data) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            received_.push_back(ReceivedMessage{message.source, std::move(message.payload)});
        }
        received_ready_.notify_one();
        return;
    }
    if (message.type == ManagerMessageType::ControlResponse) {
        std::shared_ptr<PendingRequest> pending;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto found = pending_.find(message.request_id);
            if (found == pending_.end()) { return; }
            pending = found->second;
            pending_.erase(found);
        }
        {
            std::lock_guard<std::mutex> lock(pending->mutex);
            pending->status = message.status;
            pending->payload = std::move(message.payload);
            pending->completed = true;
        }
        pending->ready.notify_one();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto selected = peers_.find(message.source);
            if (selected == peers_.end() || selected->second == channel) { return; }
        }
        RemoveChannel(channel, false);
        return;
    }
    if (!IsControlRequest(message.type)) {
        RemoveChannel(channel, true);
        return;
    }

    Metadata response;
    const auto status =
        handler_(message.type, message.protocol, message.source, message.payload, response);
    Message reply;
    reply.type = ManagerMessageType::ControlResponse;
    reply.request_id = message.request_id;
    reply.source = local_;
    reply.status = status;
    reply.payload = std::move(response);
    (void)SendMessage(channel, reply);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto selected = peers_.find(message.source);
        if (selected == peers_.end() || selected->second == channel) { return; }
    }
    RemoveChannel(channel, false);
}

void ChannelManager::SetConnected(const Endpoint& endpoint, TransportProtocol protocol,
                                  bool connected)
{
    const auto peer = endpoint.ToString();
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = peers_.find(peer);
    if (found == peers_.end()) { return; }
    if (connected) {
        found->second->connected_protocols_.insert(protocol);
    } else {
        found->second->connected_protocols_.erase(protocol);
    }
}

bool ChannelManager::IsConnected(const Endpoint& endpoint, TransportProtocol protocol) const
{
    const auto peer = endpoint.ToString();
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = peers_.find(peer);
    return found != peers_.end() && found->second->connected_protocols_.find(protocol) !=
                                        found->second->connected_protocols_.end();
}

std::vector<TransportProtocol> ChannelManager::ConnectedProtocols(const Endpoint& endpoint) const
{
    std::vector<TransportProtocol> result;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = peers_.find(endpoint.ToString());
    if (found == peers_.end()) { return result; }
    result.assign(found->second->connected_protocols_.begin(),
                  found->second->connected_protocols_.end());
    return result;
}

std::vector<ManagerID> ChannelManager::Peers() const
{
    std::vector<ManagerID> result;
    std::lock_guard<std::mutex> lock(mutex_);
    result.reserve(peers_.size());
    for (const auto& peer : peers_) { result.push_back(peer.first); }
    return result;
}

Status ChannelManager::Close(const Endpoint& endpoint)
{
    const auto peer = endpoint.ToString();
    std::shared_ptr<Channel> channel;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = peers_.find(peer);
        if (found == peers_.end()) { return Status::OK(); }
        channel = found->second;
    }
    auto completion = std::make_shared<std::promise<void>>();
    auto result = completion->get_future();
    P2P_RETURN_IF_ERROR(SubmitTask(peer,
                                   [this, channel, completion]() {
                                       RemoveChannel(channel, false);
                                       completion->set_value();
                                   }),
                        "transport manager channel close task submission failed peer={}", peer);
    result.get();
    return Status::OK();
}

void ChannelManager::RemoveChannel(const std::shared_ptr<Channel>& channel, bool cleanup_transports)
{
    ManagerID peer;
    std::vector<TransportProtocol> transports;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sockets_.erase(channel->socket_) == 0) { return; }
        (void)epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, channel->socket_, nullptr);
        peer = channel->peer_;
        if (!peer.empty()) {
            const auto found = peers_.find(peer);
            if (found != peers_.end() && found->second == channel) { peers_.erase(found); }
        }
        for (const auto protocol : channel->connected_protocols_) {
            transports.push_back(protocol);
        }
        for (auto it = pending_.begin(); it != pending_.end();) {
            if (it->second->channel.lock() != channel) {
                ++it;
                continue;
            }
            const auto request = it->second;
            it = pending_.erase(it);
            {
                std::lock_guard<std::mutex> request_lock(request->mutex);
                request->completed = true;
                request->status = Status::Error();
            }
            request->ready.notify_one();
        }
    }
    if (cleanup_transports && !peer.empty() && !transports.empty()) {
        (void)SubmitTask(peer, [this, peer, transports = std::move(transports)]() {
            for (const auto protocol : transports) {
                Metadata response;
                const auto status =
                    handler_(ManagerMessageType::DisconnectRequest, protocol, peer, {}, response);
                P2P_LOG_IF_ERROR(status,
                                 "transport manager channel cleanup failed protocol={} peer={}",
                                 static_cast<uint32_t>(protocol), peer);
            }
        });
    }
}

void ChannelManager::Shutdown()
{
    std::thread thread;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        received_ready_.notify_all();
        thread = std::move(receive_thread_);
    }
    tcp_.Stop();
    if (thread.joinable()) { thread.join(); }
    if (thread_pool_) { thread_pool_->Shutdown(); }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& item : pending_) {
        auto& request = item.second;
        {
            std::lock_guard<std::mutex> request_lock(request->mutex);
            request->completed = true;
            request->status = Status::Error();
        }
        request->ready.notify_one();
    }
    peers_.clear();
    sockets_.clear();
    pending_.clear();
    task_queues_.clear();
    received_.clear();
    thread_pool_.reset();
    if (epoll_fd_ >= 0) {
        close(epoll_fd_);
        epoll_fd_ = -1;
    }
}

}  // namespace transport
