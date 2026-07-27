#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include "core/transport.h"

namespace transport {

class TcpMessageChannel final {
public:
    TcpMessageChannel();
    ~TcpMessageChannel();

    TcpMessageChannel(const TcpMessageChannel&) = delete;
    TcpMessageChannel& operator=(const TcpMessageChannel&) = delete;

    Status Init(const Endpoint& local);
    Status Send(const Endpoint& peer, const void* data, size_t length);
    Status Receive(Endpoint& peer, Metadata& data);
    Status Shutdown();

private:
    using Socket = int;

    struct Message {
        Endpoint peer;
        Metadata data;
    };

    struct Connection {
        Socket socket = -1;
        Endpoint peer;
        std::shared_ptr<std::mutex> send_mutex;
        Metadata receive_buffer;
    };

    void CloseConnectionSocketLocked(const Connection& connection);
    void CloseSocketLocked(Socket socket);
    void CloseAllLocked();
    Status StartIoThread();
    void RunEventLoop(std::promise<Status> startup);
    void RegisterConnectionEvents(int epoll_fd, std::unordered_set<Socket>& registered);
    void HandleAcceptEvent(Socket listen_socket);
    void HandleConnectionEvent(int epoll_fd, std::unordered_set<Socket>& registered, Socket socket);
    void RemoveConnection(int epoll_fd, std::unordered_set<Socket>& registered, Socket socket);
    bool BindPeerLocked(Socket socket, const Endpoint& peer);

    Endpoint local_;
    Socket listen_socket_ = -1;
    std::mutex mutex_;
    std::condition_variable receive_cv_;
    std::deque<Message> receive_queue_;
    std::unordered_map<Socket, Connection> connections_;
    std::unordered_map<std::string, Socket> peer_sockets_;
    std::thread io_thread_;
    bool stopping_ = false;
};

}  // namespace transport
