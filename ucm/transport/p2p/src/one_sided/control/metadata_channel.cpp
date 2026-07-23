#include "control/metadata_channel.h"
#include <algorithm>
#include <arpa/inet.h>
#include <cstring>
#include <mutex>
#include <netdb.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include "common/metadata_codec.h"
#include "logger/logger.h"

namespace transport {
namespace {

using Socket = int;
constexpr Socket kInvalidSocket = -1;

struct ControlMetadataResponse {
    Status status = Status::OK();
    Metadata local_metadata;
};

constexpr int kListenBacklog = 16;

Status EncodeControlMetadataResponse(const ControlMetadataResponse& response, Metadata& out)
{
    out.clear();
    const uint32_t status =
        response.status.Success() ? 0 : (response.status == Status::InvalidParam() ? 1 : 2);
    detail::AppendU32(out, status);
    if (!detail::AppendMetadata(out, response.local_metadata)) { return Status::InvalidParam(); }
    return Status::OK();
}

Status DecodeControlMetadataResponse(const Metadata& in, ControlMetadataResponse& response)
{
    size_t offset = 0;
    uint32_t status = 0;
    if (!detail::ReadU32(in, offset, status) ||
        !detail::ReadMetadata(in, offset, response.local_metadata) || offset != in.size()) {
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
        if (sent <= 0) { return Status::Error(); }
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
        if (received <= 0) { return Status::Error(); }
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
        return Status::InvalidParam();
    }
    const uint32_t network_length = htonl(static_cast<uint32_t>(length));
    auto status = SendAll(socket, &network_length, sizeof(network_length));
    if (status != Status::OK() || length == 0) { return status; }
    return SendAll(socket, data, length);
}

Status ReceiveFrame(Socket socket, Metadata& metadata, size_t max_length)
{
    if (socket == kInvalidSocket) { return Status::InvalidParam(); }
    uint32_t network_length = 0;
    auto status = RecvAll(socket, &network_length, sizeof(network_length));
    if (status != Status::OK()) { return status; }
    const auto length = ntohl(network_length);
    if (length > max_length) { return Status::InvalidParam(); }
    metadata.assign(length, 0);
    return length == 0 ? Status::OK() : RecvAll(socket, metadata.data(), metadata.size());
}

}  // namespace

MetadataChannel::SocketHandle::SocketHandle() : socket_(kInvalidSocket) {}

MetadataChannel::SocketHandle::SocketHandle(int socket) : socket_(socket) {}

MetadataChannel::SocketHandle::~SocketHandle() { Reset(); }

MetadataChannel::SocketHandle::SocketHandle(SocketHandle&& other) noexcept
    : socket_(other.Release())
{
}

MetadataChannel::SocketHandle& MetadataChannel::SocketHandle::operator=(
    SocketHandle&& other) noexcept
{
    if (this != &other) { Reset(other.Release()); }
    return *this;
}

bool MetadataChannel::SocketHandle::Valid() const { return socket_ != kInvalidSocket; }

int MetadataChannel::SocketHandle::Get() const { return socket_; }

int MetadataChannel::SocketHandle::Release()
{
    const auto socket = socket_;
    socket_ = kInvalidSocket;
    return socket;
}

void MetadataChannel::SocketHandle::Reset(int socket)
{
    if (socket_ == socket) { return; }
    CloseSocket(socket_);
    socket_ = socket;
}

MetadataChannel::MetadataChannel() = default;

MetadataChannel::~MetadataChannel() { Close(); }

Status MetadataChannel::Init(const Endpoint& endpoint, MetadataRequestHandler handler)
{
    Close();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        metadata_request_handler_ = std::move(handler);
    }
    auto status = Listen(endpoint);
    if (status != Status::OK()) {
        Close();
        return status;
    }
    status = StartAccepting();
    if (status != Status::OK()) { Close(); }
    return status;
}

Status MetadataChannel::Listen(const Endpoint& endpoint)
{
    if (endpoint.port == 0) { return Status::InvalidParam(); }
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

Status MetadataChannel::AcceptSocket(SocketHandle& socket)
{
    if (!listen_socket_.Valid()) { return Status::InvalidParam(); }

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

Status MetadataChannel::Connect(const Endpoint& endpoint)
{
    if (endpoint.host.empty() || endpoint.port == 0) { return Status::InvalidParam(); }
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

Status MetadataChannel::ExchangeMetadata(const Endpoint& endpoint, const Metadata& metadata,
                                         Metadata& remote_metadata)
{
    MetadataChannel channel;
    auto status = channel.Connect(endpoint);
    if (status != Status::OK()) { return status; }

    status = SendFrame(channel.socket_.Get(), metadata.data(), metadata.size());
    if (status != Status::OK()) { return status; }

    Metadata response_frame;
    status = ReceiveFrame(channel.socket_.Get(), response_frame, max_receive_frame_size_);
    if (status != Status::OK()) { return status; }
    ControlMetadataResponse response;
    status = DecodeControlMetadataResponse(response_frame, response);
    if (status != Status::OK() || response.status != Status::OK()) {
        return status == Status::OK() ? response.status : status;
    }
    remote_metadata = std::move(response.local_metadata);
    return Status::OK();
}

Status MetadataChannel::StartAccepting()
{
    if (!listen_socket_.Valid()) { return Status::InvalidParam(); }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!metadata_request_handler_) { return Status::InvalidParam(); }
        if (accept_thread_.joinable()) { return Status::OK(); }
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
            if (receive_status != Status::OK()) { continue; }

            MetadataRequestHandler metadata_handler;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                metadata_handler = metadata_request_handler_;
            }
            Metadata local_metadata;
            const auto metadata_status = metadata_handler
                                             ? metadata_handler(request_frame, local_metadata)
                                             : Status::InvalidParam();
            ControlMetadataResponse metadata_response{metadata_status, std::move(local_metadata)};
            Metadata response_payload;
            if (EncodeControlMetadataResponse(metadata_response, response_payload) ==
                Status::OK()) {
                const auto send_status = SendFrame(accepted_socket.Get(), response_payload.data(),
                                                   response_payload.size());
                if (send_status != Status::OK()) { continue; }
            }
        }
    });
    return Status::OK();
}

void MetadataChannel::Close()
{
    UC_DEBUG("transport tcp close socket={} listen_socket={}", socket_.Get(), listen_socket_.Get());
    stop_accept_.store(true, std::memory_order_relaxed);
    listen_socket_.Reset();
    if (accept_thread_.joinable()) { accept_thread_.join(); }
    socket_.Reset();
}

}  // namespace transport
