#include "two_sided/tcp/tcp_message_channel.h"
#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <future>
#include <mutex>
#include <netdb.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "common/metadata_codec.h"

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace transport {
namespace {

using Socket = int;
constexpr Socket kInvalidSocket = -1;
constexpr uint32_t kMaxFrameSize = 4 * 1024 * 1024;
constexpr int kListenBacklog = 16;
constexpr size_t kEncodedEndpointOverhead = sizeof(uint32_t) + sizeof(uint16_t);

void CloseSocket(Socket socket)
{
    if (socket != kInvalidSocket) {
        ::shutdown(socket, SHUT_RDWR);
        close(socket);
    }
}

Status EncodeMessage(const Endpoint& local, const void* data, size_t length, Metadata& frame)
{
    frame.clear();
    if (local.host.size() > UINT32_MAX) { return Status::InvalidArgument; }
    const auto header_size = kEncodedEndpointOverhead + local.host.size();
    if (header_size > kMaxFrameSize || length > kMaxFrameSize - header_size) {
        return Status::InvalidArgument;
    }
    if (!detail::AppendString(frame, local.host) || !detail::AppendU16(frame, local.port)) {
        return Status::InvalidArgument;
    }
    if (length != 0) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        frame.insert(frame.end(), bytes, bytes + length);
    }
    return Status::Ok;
}

bool DecodeMessage(const Metadata& frame, Endpoint& peer, Metadata& data)
{
    size_t offset = 0;
    if (!detail::ReadString(frame, offset, peer.host) ||
        !detail::ReadU16(frame, offset, peer.port)) {
        return false;
    }
    data.assign(frame.begin() + static_cast<ptrdiff_t>(offset), frame.end());
    return true;
}

Status SendAll(Socket socket, const void* data, size_t length)
{
    const auto* cursor = static_cast<const char*>(data);
    while (length > 0) {
        const auto chunk = static_cast<int>(std::min<size_t>(length, 64 * 1024));
        const int sent = ::send(socket, cursor, chunk, MSG_NOSIGNAL);
        if (sent <= 0) { return Status::Failed; }
        cursor += sent;
        length -= static_cast<size_t>(sent);
    }
    return Status::Ok;
}

Status SendFrame(Socket socket, const Metadata& data)
{
    if (socket == kInvalidSocket || data.size() > UINT32_MAX) { return Status::InvalidArgument; }
    const uint32_t network_length = htonl(static_cast<uint32_t>(data.size()));
    auto status = SendAll(socket, &network_length, sizeof(network_length));
    if (status != Status::Ok || data.empty()) { return status; }
    return SendAll(socket, data.data(), data.size());
}

Status ConnectSocket(const Endpoint& endpoint, Socket& socket)
{
    if (endpoint.host.empty() || endpoint.port == 0) { return Status::InvalidArgument; }

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* results = nullptr;
    const auto port = std::to_string(endpoint.port);
    if (getaddrinfo(endpoint.host.c_str(), port.c_str(), &hints, &results) != 0) {
        return Status::Failed;
    }

    Status status = Status::Failed;
    for (auto* item = results; item != nullptr; item = item->ai_next) {
        const auto candidate = ::socket(item->ai_family, item->ai_socktype, item->ai_protocol);
        if (candidate == kInvalidSocket) { continue; }
        if (::connect(candidate, item->ai_addr, static_cast<int>(item->ai_addrlen)) == 0) {
            socket = candidate;
            status = Status::Ok;
            break;
        }
        CloseSocket(candidate);
    }
    freeaddrinfo(results);
    return status;
}

Status ListenSocket(const Endpoint& endpoint, int backlog, Socket& socket)
{
    if (endpoint.port == 0 || backlog <= 0) { return Status::InvalidArgument; }

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* results = nullptr;
    const auto port = std::to_string(endpoint.port);
    const char* host = endpoint.host.empty() ? nullptr : endpoint.host.c_str();
    if (getaddrinfo(host, port.c_str(), &hints, &results) != 0) { return Status::Failed; }

    Status status = Status::Failed;
    for (auto* item = results; item != nullptr; item = item->ai_next) {
        const auto candidate = ::socket(item->ai_family, item->ai_socktype, item->ai_protocol);
        if (candidate == kInvalidSocket) { continue; }
        int yes = 1;
        if (setsockopt(candidate, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes),
                       sizeof(yes)) != 0) {
            CloseSocket(candidate);
            continue;
        }
        if (::bind(candidate, item->ai_addr, static_cast<int>(item->ai_addrlen)) == 0 &&
            ::listen(candidate, backlog) == 0) {
            socket = candidate;
            status = Status::Ok;
            break;
        }
        CloseSocket(candidate);
    }
    freeaddrinfo(results);
    return status;
}

Status ReceiveAvailableFrames(Socket socket, Metadata& buffer, std::vector<Metadata>& frames,
                              uint32_t max_frame_size, bool& closed)
{
    closed = false;
    char chunk[64 * 1024];
    for (;;) {
        const auto received = ::recv(socket, chunk, sizeof(chunk), MSG_DONTWAIT);
        if (received > 0) {
            buffer.insert(buffer.end(), chunk, chunk + received);
            continue;
        }
        if (received == 0) {
            closed = true;
            return Status::Ok;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) { break; }
        if (errno == EINTR) { continue; }
        return Status::Failed;
    }

    size_t offset = 0;
    while (buffer.size() - offset >= sizeof(uint32_t)) {
        uint32_t network_length = 0;
        std::copy_n(buffer.data() + offset, sizeof(network_length),
                    reinterpret_cast<uint8_t*>(&network_length));
        const auto length = ntohl(network_length);
        if (length > max_frame_size) { return Status::InvalidArgument; }
        if (buffer.size() - offset - sizeof(uint32_t) < length) { break; }
        const auto payload_begin = offset + sizeof(uint32_t);
        frames.emplace_back(buffer.begin() + static_cast<ptrdiff_t>(payload_begin),
                            buffer.begin() + static_cast<ptrdiff_t>(payload_begin + length));
        offset = payload_begin + length;
    }
    if (offset != 0) {
        buffer.erase(buffer.begin(), buffer.begin() + static_cast<ptrdiff_t>(offset));
    }
    return Status::Ok;
}

}  // namespace

void TcpMessageChannel::CloseConnectionSocketLocked(const Connection& connection)
{
    if (connection.send_mutex) {
        std::lock_guard<std::mutex> send_lock(*connection.send_mutex);
        CloseSocket(connection.socket);
        return;
    }
    CloseSocket(connection.socket);
}

void TcpMessageChannel::CloseSocketLocked(Socket socket)
{
    if (socket == kInvalidSocket) { return; }
    auto connection_it = connections_.find(socket);
    if (connection_it != connections_.end()) {
        if (!connection_it->second.peer.host.empty() && connection_it->second.peer.port != 0) {
            const auto peer_it = peer_sockets_.find(connection_it->second.peer.ToString());
            if (peer_it != peer_sockets_.end() && peer_it->second == socket) {
                peer_sockets_.erase(peer_it);
            }
        }
        CloseConnectionSocketLocked(connection_it->second);
        connections_.erase(connection_it);
        return;
    }
    CloseSocket(socket);
}

void TcpMessageChannel::CloseAllLocked()
{
    CloseSocket(listen_socket_);
    listen_socket_ = kInvalidSocket;
    for (const auto& item : connections_) { CloseConnectionSocketLocked(item.second); }
    connections_.clear();
    peer_sockets_.clear();
}

TcpMessageChannel::TcpMessageChannel() = default;

TcpMessageChannel::~TcpMessageChannel()
{
    if (Shutdown() != Status::Ok) {}
}

Status TcpMessageChannel::StartIoThread()
{
    if (io_thread_.joinable()) { return Status::Ok; }
    std::promise<Status> startup;
    auto startup_result = startup.get_future();
    io_thread_ = std::thread(
        [this, startup = std::move(startup)]() mutable { RunEventLoop(std::move(startup)); });
    const auto status = startup_result.get();
    if (status != Status::Ok && io_thread_.joinable()) { io_thread_.join(); }
    return status;
}

void TcpMessageChannel::RunEventLoop(std::promise<Status> startup)
{
    const int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (epoll_fd < 0) {
        startup.set_value(Status::Failed);
        return;
    }

    Socket active_listen_socket = kInvalidSocket;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_listen_socket = listen_socket_;
    }
    if (active_listen_socket == kInvalidSocket) {
        ::close(epoll_fd);
        startup.set_value(Status::Failed);
        return;
    }

    epoll_event event{};
    event.events = EPOLLIN | EPOLLHUP | EPOLLERR;
    event.data.fd = active_listen_socket;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, active_listen_socket, &event) != 0) {
        ::close(epoll_fd);
        startup.set_value(Status::Failed);
        return;
    }

    startup.set_value(Status::Ok);
    std::unordered_set<Socket> registered;
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) { break; }
        }
        RegisterConnectionEvents(epoll_fd, registered);

        const int max_events = static_cast<int>(std::max<size_t>(registered.size() + 1, 1));
        std::vector<epoll_event> events(static_cast<size_t>(max_events));
        const int ready = epoll_wait(epoll_fd, events.data(), max_events, 100);
        if (ready <= 0) { continue; }

        for (int index = 0; index < ready; ++index) {
            const Socket socket = events[index].data.fd;
            if ((events[index].events & (EPOLLIN | EPOLLHUP | EPOLLERR)) == 0) { continue; }
            if (socket == active_listen_socket) {
                HandleAcceptEvent(active_listen_socket);
            } else {
                HandleConnectionEvent(epoll_fd, registered, socket);
            }
        }
    }

    for (const auto socket : registered) {
        if (epoll_ctl(epoll_fd, EPOLL_CTL_DEL, socket, nullptr) != 0) { continue; }
    }
    ::close(epoll_fd);
}

void TcpMessageChannel::RegisterConnectionEvents(int epoll_fd,
                                                 std::unordered_set<Socket>& registered)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& item : connections_) {
        const Socket socket = item.first;
        if (registered.find(socket) != registered.end()) { continue; }
        epoll_event event{};
        event.events = EPOLLIN | EPOLLHUP | EPOLLERR;
        event.data.fd = socket;
        if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, socket, &event) == 0) { registered.insert(socket); }
    }
}

void TcpMessageChannel::HandleAcceptEvent(Socket active_listen_socket)
{
    const auto accepted = ::accept(active_listen_socket, nullptr, nullptr);
    if (accepted == kInvalidSocket) { return; }

    Connection connection;
    connection.socket = accepted;
    connection.send_mutex = std::make_shared<std::mutex>();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            CloseSocket(accepted);
            return;
        }
        connections_.emplace(accepted, std::move(connection));
    }
}

void TcpMessageChannel::HandleConnectionEvent(int epoll_fd, std::unordered_set<Socket>& registered,
                                              Socket socket)
{
    std::vector<Metadata> frames;
    bool closed = false;
    Status receive_status = Status::Ok;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = connections_.find(socket);
        if (it == connections_.end()) { return; }
        receive_status = ReceiveAvailableFrames(socket, it->second.receive_buffer, frames,
                                                kMaxFrameSize, closed);
    }
    if (receive_status != Status::Ok || closed) {
        RemoveConnection(epoll_fd, registered, socket);
        return;
    }

    bool drop_socket = false;
    for (const auto& frame : frames) {
        Message message;
        if (!DecodeMessage(frame, message.peer, message.data)) {
            drop_socket = true;
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto connection_it = connections_.find(socket);
            if (connection_it == connections_.end()) {
                drop_socket = true;
            } else {
                const auto peer_bound = !connection_it->second.peer.host.empty() &&
                                        connection_it->second.peer.port != 0;
                if (!peer_bound && !BindPeerLocked(socket, message.peer)) { drop_socket = true; }
            }
            receive_queue_.emplace_back(std::move(message));
        }
        receive_cv_.notify_one();
    }
    if (drop_socket) { RemoveConnection(epoll_fd, registered, socket); }
}

void TcpMessageChannel::RemoveConnection(int epoll_fd, std::unordered_set<Socket>& registered,
                                         Socket socket)
{
    epoll_ctl(epoll_fd, EPOLL_CTL_DEL, socket, nullptr);
    registered.erase(socket);
    std::lock_guard<std::mutex> lock(mutex_);
    CloseSocketLocked(socket);
}

bool TcpMessageChannel::BindPeerLocked(Socket socket, const Endpoint& peer)
{
    auto connection_it = connections_.find(socket);
    if (connection_it == connections_.end()) { return false; }

    const auto peer_id = peer.ToString();
    auto existing = peer_sockets_.find(peer_id);
    connection_it->second.peer = peer;
    if (existing != peer_sockets_.end() &&
        connections_.find(existing->second) == connections_.end()) {
        peer_sockets_.erase(existing);
        existing = peer_sockets_.end();
    }
    if (existing == peer_sockets_.end()) { peer_sockets_[peer_id] = socket; }
    return true;
}

Status TcpMessageChannel::Init(const Endpoint& local)
{
    if (local.host.empty() || local.port == 0) { return Status::InvalidArgument; }
    const auto shutdown_status = Shutdown();
    if (shutdown_status != Status::Ok) { return shutdown_status; }

    Socket listen_socket = kInvalidSocket;
    auto status = ListenSocket(local, kListenBacklog, listen_socket);
    if (status != Status::Ok) { return status; }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        local_ = local;
        listen_socket_ = listen_socket;
        stopping_ = false;
    }

    status = StartIoThread();
    if (status != Status::Ok) { Shutdown(); }
    return status;
}

Status TcpMessageChannel::Send(const Endpoint& peer, const void* data, size_t length)
{
    if (peer.host.empty() || peer.port == 0 || (data == nullptr && length != 0) ||
        length > UINT32_MAX) {
        return Status::InvalidArgument;
    }

    Endpoint local;
    std::string peer_id;
    Socket socket = kInvalidSocket;
    std::shared_ptr<std::mutex> send_mutex;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || listen_socket_ == kInvalidSocket) { return Status::Failed; }
        local = local_;
        peer_id = peer.ToString();
        auto peer_it = peer_sockets_.find(peer_id);
        if (peer_it != peer_sockets_.end()) {
            const auto connection_it = connections_.find(peer_it->second);
            if (connection_it != connections_.end()) {
                socket = connection_it->second.socket;
                send_mutex = connection_it->second.send_mutex;
            } else {
                peer_sockets_.erase(peer_it);
            }
        }
    }

    if (socket == kInvalidSocket) {
        Socket connected = kInvalidSocket;
        auto status = ConnectSocket(peer, connected);
        if (status != Status::Ok) { return status; }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                CloseSocket(connected);
                return Status::Failed;
            }
            Connection connection;
            connection.socket = connected;
            connection.peer = peer;
            connection.send_mutex = std::make_shared<std::mutex>();
            socket = connected;
            send_mutex = connection.send_mutex;
            connections_.emplace(connected, std::move(connection));
            peer_sockets_[peer_id] = connected;
        }
    }

    if (socket == kInvalidSocket || !send_mutex) { return Status::Failed; }

    Metadata frame;
    auto status = EncodeMessage(local, data, length, frame);
    if (status != Status::Ok) { return status; }
    {
        std::lock_guard<std::mutex> send_lock(*send_mutex);
        status = SendFrame(socket, frame);
    }
    if (status != Status::Ok) {
        std::lock_guard<std::mutex> lock(mutex_);
        CloseSocketLocked(socket);
    }
    return status;
}

Status TcpMessageChannel::Receive(Endpoint& peer, Metadata& data)
{
    std::unique_lock<std::mutex> lock(mutex_);
    receive_cv_.wait(lock, [this]() { return stopping_ || !receive_queue_.empty(); });
    if (receive_queue_.empty()) { return Status::Failed; }
    auto message = std::move(receive_queue_.front());
    receive_queue_.pop_front();
    peer = std::move(message.peer);
    data = std::move(message.data);
    return Status::Ok;
}

Status TcpMessageChannel::Shutdown()
{
    std::thread io_thread;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        receive_cv_.notify_all();
        CloseAllLocked();
        receive_queue_.clear();
        io_thread = std::move(io_thread_);
    }
    if (io_thread.joinable()) { io_thread.join(); }
    return Status::Ok;
}

}  // namespace transport
