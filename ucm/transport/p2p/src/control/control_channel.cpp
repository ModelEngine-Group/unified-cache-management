#include "control/control_channel.h"
#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <netdb.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include "common/binary_codec.h"
#include "logger/logger.h"

namespace transport {
namespace {

using Socket = int;
constexpr Socket kInvalidSocket = -1;

struct ControlResponse {
    Status status = Status::OK();
    Metadata payload;
};

constexpr int kListenBacklog = 16;

Status EncodeControlResponse(const ControlResponse& response, Metadata& out)
{
    out.clear();
    const uint32_t status =
        response.status.Success() ? 0 : (response.status == Status::InvalidParam() ? 1 : 2);
    detail::AppendU32(out, status);
    if (!detail::AppendBytes(out, response.payload)) { return Status::InvalidParam(); }
    return Status::OK();
}

Status DecodeControlResponse(const Metadata& in, ControlResponse& response)
{
    size_t offset = 0;
    uint32_t status = 0;
    if (!detail::ReadU32(in, offset, status) || !detail::ReadBytes(in, offset, response.payload) ||
        offset != in.size()) {
        return Status::InvalidParam();
    }
    switch (status) {
        case 0: response.status = Status::OK(); break;
        case 1: response.status = Status::InvalidParam(); break;
        case 2: response.status = Status::Error(); break;
        default: return Status::InvalidParam();
    }
    return Status::OK();
}

Status SendAll(Socket socket, const void* data, size_t length)
{
    const auto* cursor = static_cast<const char*>(data);
    while (length > 0) {
        const auto chunk = static_cast<int>(std::min<size_t>(length, 64 * 1024));
        const int sent = send(socket, cursor, chunk, 0);
        if (sent <= 0) {
            UC_ERROR("transport tcp send failed socket={} remaining={} errno={} error={}", socket,
                     length, errno, std::strerror(errno));
            return Status::Error();
        }
        cursor += sent;
        length -= static_cast<size_t>(sent);
    }
    return Status::OK();
}

Status RecvAll(Socket socket, void* data, size_t length)
{
    auto* cursor = static_cast<char*>(data);
    while (length > 0) {
        const auto chunk = static_cast<int>(std::min<size_t>(length, 64 * 1024));
        const int received = recv(socket, cursor, chunk, 0);
        if (received == 0) {
            UC_ERROR("transport tcp peer closed connection socket={} remaining={}", socket, length);
            return Status::Error();
        }
        if (received < 0) {
            UC_ERROR(
                "transport tcp receive failed socket={} remaining={} result={} errno={} error={}",
                socket, length, received, errno, std::strerror(errno));
            return Status::Error();
        }
        cursor += received;
        length -= static_cast<size_t>(received);
    }
    return Status::OK();
}

void CloseSocket(Socket socket)
{
    if (socket != kInvalidSocket) {
        if (::shutdown(socket, SHUT_RDWR) != 0) {
            UC_DEBUG("transport tcp shutdown socket={} failed", socket);
        }
        if (close(socket) != 0) { UC_DEBUG("transport tcp close socket={} failed", socket); }
    }
}

Status SendFrame(Socket socket, const void* data, size_t length)
{
    if (socket == kInvalidSocket || (data == nullptr && length != 0) || length > UINT32_MAX) {
        UC_ERROR("transport tcp send frame invalid socket={} data={} length={}", socket,
                 static_cast<const void*>(data), length);
        return Status::InvalidParam();
    }
    const uint32_t network_length = htonl(static_cast<uint32_t>(length));
    auto status = SendAll(socket, &network_length, sizeof(network_length));
    if (status != Status::OK() || length == 0) { return status; }
    return SendAll(socket, data, length);
}

Status ReceiveFrame(Socket socket, Metadata& metadata, size_t max_length)
{
    if (socket == kInvalidSocket) {
        UC_ERROR("transport tcp receive frame invalid socket={}", socket);
        return Status::InvalidParam();
    }
    uint32_t network_length = 0;
    auto status = RecvAll(socket, &network_length, sizeof(network_length));
    if (status != Status::OK()) { return status; }
    const auto length = ntohl(network_length);
    if (length > max_length) {
        UC_ERROR("transport tcp receive frame too large socket={} length={} limit={}", socket,
                 length, max_length);
        return Status::InvalidParam();
    }
    metadata.assign(length, 0);
    return length == 0 ? Status::OK() : RecvAll(socket, metadata.data(), metadata.size());
}

}  // namespace

ControlChannel::SocketHandle::SocketHandle() : socket_(kInvalidSocket) {}

ControlChannel::SocketHandle::SocketHandle(int socket) : socket_(socket) {}

ControlChannel::SocketHandle::~SocketHandle() { Reset(); }

ControlChannel::SocketHandle::SocketHandle(SocketHandle&& other) noexcept : socket_(other.Release())
{
}

ControlChannel::SocketHandle& ControlChannel::SocketHandle::operator=(SocketHandle&& other) noexcept
{
    if (this != &other) { Reset(other.Release()); }
    return *this;
}

bool ControlChannel::SocketHandle::Valid() const { return socket_ != kInvalidSocket; }

int ControlChannel::SocketHandle::Get() const { return socket_; }

int ControlChannel::SocketHandle::Release()
{
    const auto socket = socket_;
    socket_ = kInvalidSocket;
    return socket;
}

void ControlChannel::SocketHandle::Reset(int socket)
{
    if (socket_ == socket) { return; }
    CloseSocket(socket_);
    socket_ = socket;
}

ControlChannel::ControlChannel() = default;

ControlChannel::~ControlChannel() { Close(); }

Status ControlChannel::Init(const Endpoint& endpoint, RequestHandler handler)
{
    UC_DEBUG("transport control init begin endpoint={}:{}", endpoint.host, endpoint.port);
    Close();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        request_handler_ = std::move(handler);
    }
    auto status = Listen(endpoint);
    if (status != Status::OK()) {
        UC_ERROR("transport control listen initialization failed endpoint={}:{} status={}",
                 endpoint.host, endpoint.port, status.Underlying());
        Close();
        return status;
    }
    status = StartAccepting();
    if (status != Status::OK()) {
        UC_ERROR("transport control accept initialization failed endpoint={}:{} status={}",
                 endpoint.host, endpoint.port, status.Underlying());
        Close();
    } else {
        UC_DEBUG("transport control init completed endpoint={}:{}", endpoint.host, endpoint.port);
    }
    return status;
}

Status ControlChannel::Listen(const Endpoint& endpoint)
{
    if (endpoint.port == 0) {
        UC_ERROR("transport tcp listen invalid endpoint={}:{}", endpoint.host, endpoint.port);
        return Status::InvalidParam();
    }
    UC_DEBUG("transport tcp listen begin endpoint={}:{} backlog={}", endpoint.host, endpoint.port,
             kListenBacklog);
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* results = nullptr;
    const auto port = std::to_string(endpoint.port);
    const char* host = endpoint.host.empty() ? nullptr : endpoint.host.c_str();
    if (getaddrinfo(host, port.c_str(), &hints, &results) != 0) {
        UC_ERROR("transport tcp listen getaddrinfo failed endpoint={}:{}", endpoint.host,
                 endpoint.port);
        return Status::Error();
    }

    Status status = Status::Error();
    for (auto* item = results; item != nullptr; item = item->ai_next) {
        SocketHandle candidate(socket(item->ai_family, item->ai_socktype, item->ai_protocol));
        if (!candidate.Valid()) { continue; }
        int yes = 1;
        if (setsockopt(candidate.Get(), SOL_SOCKET, SO_REUSEADDR,
                       reinterpret_cast<const char*>(&yes), sizeof(yes)) != 0) {
            continue;
        }
        if (bind(candidate.Get(), item->ai_addr, static_cast<int>(item->ai_addrlen)) == 0 &&
            ::listen(candidate.Get(), kListenBacklog) == 0) {
            const auto socket = candidate.Get();
            listen_socket_.Reset(candidate.Release());
            endpoint_ = endpoint;
            status = Status::OK();
            UC_DEBUG("transport tcp listen ok endpoint={}:{} socket={}", endpoint.host,
                     endpoint.port, socket);
            break;
        }
    }

    freeaddrinfo(results);
    if (status != Status::OK()) {
        UC_ERROR("transport tcp listen failed endpoint={}:{}", endpoint.host, endpoint.port);
    }
    return status;
}

Status ControlChannel::AcceptSocket(SocketHandle& socket)
{
    if (!listen_socket_.Valid()) {
        UC_ERROR("transport tcp accept failed: listen socket is invalid");
        return Status::InvalidParam();
    }

    const auto accepted = ::accept(listen_socket_.Get(), nullptr, nullptr);
    if (accepted == kInvalidSocket) {
        if (stop_accept_.load(std::memory_order_relaxed)) {
            UC_DEBUG("transport tcp accept stopped");
            return Status::Error();
        }
        UC_ERROR("transport tcp accept failed");
        return Status::Error();
    }

    socket.Reset(accepted);
    UC_DEBUG("transport tcp accept ok socket={}", accepted);
    return Status::OK();
}

Status ControlChannel::Connect(const Endpoint& endpoint)
{
    if (endpoint.host.empty() || endpoint.port == 0) {
        UC_ERROR("transport tcp connect invalid endpoint={}:{}", endpoint.host, endpoint.port);
        return Status::InvalidParam();
    }
    UC_DEBUG("transport tcp connect begin endpoint={}:{}", endpoint.host, endpoint.port);
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* results = nullptr;
    const auto port = std::to_string(endpoint.port);
    if (getaddrinfo(endpoint.host.c_str(), port.c_str(), &hints, &results) != 0) {
        UC_ERROR("transport tcp connect getaddrinfo failed endpoint={}:{}", endpoint.host,
                 endpoint.port);
        return Status::Error();
    }

    Status status = Status::Error();
    for (auto* item = results; item != nullptr; item = item->ai_next) {
        SocketHandle candidate(socket(item->ai_family, item->ai_socktype, item->ai_protocol));
        if (!candidate.Valid()) { continue; }
        if (::connect(candidate.Get(), item->ai_addr, static_cast<int>(item->ai_addrlen)) == 0) {
            const auto socket = candidate.Get();
            socket_.Reset(candidate.Release());
            status = Status::OK();
            UC_DEBUG("transport tcp connect ok endpoint={}:{} socket={}", endpoint.host,
                     endpoint.port, socket);
            break;
        }
    }

    freeaddrinfo(results);
    if (status != Status::OK()) {
        UC_ERROR("transport tcp connect failed endpoint={}:{}", endpoint.host, endpoint.port);
    }
    return status;
}

Status ControlChannel::Request(const Endpoint& endpoint, const Metadata& request,
                               Metadata& response)
{
    UC_DEBUG("transport control request begin peer={}:{} bytes={}", endpoint.host, endpoint.port,
             request.size());
    ControlChannel channel;
    auto status = channel.Connect(endpoint);
    if (status != Status::OK()) {
        UC_ERROR("transport control request connect failed peer={}:{} status={}", endpoint.host,
                 endpoint.port, status.Underlying());
        return status;
    }

    status = SendFrame(channel.socket_.Get(), request.data(), request.size());
    if (status != Status::OK()) {
        UC_ERROR("transport control request send failed peer={}:{} socket={} status={}",
                 endpoint.host, endpoint.port, channel.socket_.Get(), status.Underlying());
        return status;
    }
    UC_DEBUG("transport control request sent peer={}:{} socket={} bytes={}", endpoint.host,
             endpoint.port, channel.socket_.Get(), request.size());

    Metadata response_frame;
    status = ReceiveFrame(channel.socket_.Get(), response_frame, max_receive_frame_size_);
    if (status != Status::OK()) {
        UC_ERROR("transport control response receive failed peer={}:{} socket={} status={}",
                 endpoint.host, endpoint.port, channel.socket_.Get(), status.Underlying());
        return status;
    }
    ControlResponse decoded_response;
    status = DecodeControlResponse(response_frame, decoded_response);
    if (status != Status::OK()) {
        UC_ERROR("transport control response decode failed peer={}:{} socket={} bytes={} status={}",
                 endpoint.host, endpoint.port, channel.socket_.Get(), response_frame.size(),
                 status.Underlying());
        return status;
    }
    if (decoded_response.status != Status::OK()) {
        UC_ERROR("transport control peer rejected request peer={}:{} socket={} status={}",
                 endpoint.host, endpoint.port, channel.socket_.Get(),
                 decoded_response.status.Underlying());
        return decoded_response.status;
    }
    response = std::move(decoded_response.payload);
    UC_DEBUG("transport control request completed peer={}:{} socket={} response_bytes={}",
             endpoint.host, endpoint.port, channel.socket_.Get(), response.size());
    return Status::OK();
}

Status ControlChannel::StartAccepting()
{
    if (!listen_socket_.Valid()) {
        UC_ERROR("transport tcp start accepting failed: listen socket is invalid");
        return Status::InvalidParam();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!request_handler_) {
            UC_ERROR("transport tcp start accepting failed: request handler is empty");
            return Status::InvalidParam();
        }
        if (accept_thread_.joinable()) {
            UC_DEBUG("transport tcp accept thread already running socket={}", listen_socket_.Get());
            return Status::OK();
        }
        stop_accept_.store(false, std::memory_order_relaxed);
    }

    accept_thread_ = std::thread([this]() {
        while (!stop_accept_.load(std::memory_order_relaxed)) {
            SocketHandle accepted_socket;
            const auto status = AcceptSocket(accepted_socket);
            if (status != Status::OK()) { continue; }

            Metadata request_frame;
            const auto receive_status =
                ReceiveFrame(accepted_socket.Get(), request_frame, max_receive_frame_size_);
            if (receive_status != Status::OK()) {
                UC_ERROR("transport control request receive failed socket={} status={}",
                         accepted_socket.Get(), receive_status.Underlying());
                continue;
            }
            UC_DEBUG("transport control request received socket={} bytes={}", accepted_socket.Get(),
                     request_frame.size());

            RequestHandler handler;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                handler = request_handler_;
            }
            Metadata response;
            const auto request_status =
                handler ? handler(request_frame, response) : Status::InvalidParam();
            if (request_status != Status::OK()) {
                UC_ERROR("transport control request handler failed socket={} status={}",
                         accepted_socket.Get(), request_status.Underlying());
            } else {
                UC_DEBUG("transport control request handled socket={} response_bytes={}",
                         accepted_socket.Get(), response.size());
            }
            ControlResponse control_response{request_status, std::move(response)};
            Metadata response_payload;
            const auto encode_status = EncodeControlResponse(control_response, response_payload);
            if (encode_status != Status::OK()) {
                UC_ERROR("transport control response encode failed socket={} status={}",
                         accepted_socket.Get(), encode_status.Underlying());
                continue;
            }
            const auto send_status =
                SendFrame(accepted_socket.Get(), response_payload.data(), response_payload.size());
            if (send_status != Status::OK()) {
                UC_ERROR("transport control response send failed socket={} bytes={} status={}",
                         accepted_socket.Get(), response_payload.size(), send_status.Underlying());
                continue;
            }
            UC_DEBUG("transport control response sent socket={} bytes={} status={}",
                     accepted_socket.Get(), response_payload.size(), request_status.Underlying());
        }
    });
    return Status::OK();
}

void ControlChannel::Close()
{
    UC_DEBUG("transport tcp close socket={} listen_socket={}", socket_.Get(), listen_socket_.Get());
    stop_accept_.store(true, std::memory_order_relaxed);
    listen_socket_.Reset();
    if (accept_thread_.joinable()) { accept_thread_.join(); }
    socket_.Reset();
}

}  // namespace transport
