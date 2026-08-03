#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include "core/transport.h"

namespace transport {

class ControlChannel final {
public:
    using RequestHandler = std::function<Status(const Metadata& request, Metadata& response)>;

    ControlChannel();
    ~ControlChannel();

    ControlChannel(const ControlChannel&) = delete;
    ControlChannel& operator=(const ControlChannel&) = delete;
    ControlChannel(ControlChannel&&) = delete;
    ControlChannel& operator=(ControlChannel&&) = delete;

    Status Init(const Endpoint& endpoint, RequestHandler handler);
    Status Request(const Endpoint& endpoint, const Metadata& request, Metadata& response);
    void Close();

private:
    class SocketHandle final {
    public:
        SocketHandle();
        explicit SocketHandle(int socket);
        ~SocketHandle();

        SocketHandle(const SocketHandle&) = delete;
        SocketHandle& operator=(const SocketHandle&) = delete;
        SocketHandle(SocketHandle&& other) noexcept;
        SocketHandle& operator=(SocketHandle&& other) noexcept;

        bool Valid() const;
        int Get() const;
        int Release();
        void Reset(int socket = -1);

    private:
        int socket_;
    };

    Status Listen(const Endpoint& endpoint);
    Status StartAccepting();
    Status Connect(const Endpoint& endpoint);
    Status AcceptSocket(SocketHandle& socket);

    Endpoint endpoint_;
    SocketHandle listen_socket_;
    SocketHandle socket_;
    mutable std::mutex mutex_;
    std::thread accept_thread_;
    std::atomic<bool> stop_accept_{false};
    uint32_t max_receive_frame_size_ = 4 * 1024 * 1024;
    RequestHandler request_handler_;
};

}  // namespace transport
