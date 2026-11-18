#include <gtest/gtest.h>
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#include "view/m800_view_slave_server.h"

namespace kv::test {
namespace {

class RawResponseServer {
public:
    explicit RawResponseServer(std::string response, int responseDelayMs = 0)
        : response_(std::move(response)), responseDelayMs_(responseDelayMs)
    {
        listenFd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (listenFd_ < 0) { throw std::runtime_error("failed to create raw server socket"); }
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = 0;
        if (bind(listenFd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
            listen(listenFd_, 1) != 0) {
            close(listenFd_);
            throw std::runtime_error("failed to bind raw server");
        }
        socklen_t size = sizeof(address);
        if (getsockname(listenFd_, reinterpret_cast<sockaddr*>(&address), &size) != 0) {
            close(listenFd_);
            throw std::runtime_error("failed to get raw server port");
        }
        port_ = ntohs(address.sin_port);
        worker_ = std::thread([this] { ServeOne(); });
    }

    ~RawResponseServer()
    {
        shutdown(listenFd_, SHUT_RDWR);
        close(listenFd_);
        if (worker_.joinable()) { worker_.join(); }
    }

    std::uint16_t Port() const { return port_; }

private:
    void ServeOne()
    {
        const int client = accept(listenFd_, nullptr, nullptr);
        if (client < 0) { return; }
        char request[2048];
        recv(client, request, sizeof(request), 0);
        if (responseDelayMs_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(responseDelayMs_));
        }
        std::size_t sent{0};
        while (sent < response_.size()) {
            const auto result = send(client, response_.data() + sent, response_.size() - sent,
                                     MSG_NOSIGNAL);
            if (result <= 0) { break; }
            sent += static_cast<std::size_t>(result);
        }
        shutdown(client, SHUT_RDWR);
        close(client);
    }

    int listenFd_{-1};
    std::uint16_t port_{0};
    std::string response_;
    int responseDelayMs_{0};
    std::thread worker_;
};

std::string ValidText(std::uint64_t epoch)
{
    const auto value = std::to_string(epoch);
    return "globalEpoch=" + value + "\nnsCount=1\nns=1\nviewEpoch=" + value +
           "\nviewId=1\nasuIds=1\n"
           "asuInfo.1=protocol=uboe,port=6000,local.comm_id=10.0.0.1\n";
}

TEST(M800ViewSlaveHttpNegativeTest, RejectsMalformedResponsesWithoutChangingUniverse)
{
    const auto validBody = ValidText(10);
    const std::vector<std::pair<std::string, StatusCode>> cases = {
        {"HTTP/1.1 304 Not Modified\r\nConnection: close\r\n\r\n",
         StatusCode::INTERNAL_ERROR},
        {"HTTP/1.1 200OK\r\nContent-Length: 0\r\n\r\n", StatusCode::INTERNAL_ERROR},
        {"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
         StatusCode::INTERNAL_ERROR},
        {"HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(validBody.size() + 10) +
             "\r\n\r\n" + validBody,
         StatusCode::CONNECTION_ERROR},
        {"HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(validBody.size() - 1) +
             "\r\n\r\n" + validBody,
         StatusCode::INTERNAL_ERROR},
    };

    for (const auto& [response, expectedCode] : cases) {
        RawResponseServer server(response);
        M800ViewServerConfig config;
        config.role = M800ViewRole::SLAVE;
        config.masterAddrs = {"127.0.0.1:" + std::to_string(server.Port())};
        config.httpTimeoutMs = 300;
        auto worker = CreateM800ViewSlaveServer(config);
        GlobalView initial;
        initial.viewEpoch = 8;
        initial.viewId = 9;
        auto universe = CreateViewUniverse({{9, initial}}, 8);
        const auto status = worker->GetAllViews(*universe);
        EXPECT_EQ(status.code, expectedCode) << response;
        EXPECT_EQ(universe->GetGlobalEpoch(), 8) << response;
        EXPECT_EQ(universe->GetAllNsIds().size(), 1) << response;
        worker->Stop();
    }
}

TEST(M800ViewSlaveHttpNegativeTest, StopInterruptsBlockedRead)
{
    RawResponseServer server("", 500);
    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(server.Port())};
    config.httpTimeoutMs = 2000;
    auto worker = CreateM800ViewSlaveServer(config);
    auto universe = CreateViewUniverse();
    Status fetchStatus;
    std::thread fetcher([&] { fetchStatus = worker->GetAllViews(*universe); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const auto begin = std::chrono::steady_clock::now();
    worker->Stop();
    fetcher.join();
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(1));
    EXPECT_EQ(fetchStatus.code, StatusCode::CANCELED);
}

TEST(M800ViewSlaveHttpNegativeTest, AllMasterAddressesShareOneDeadline)
{
    RawResponseServer first("", 500);
    RawResponseServer second("", 500);
    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(first.Port()),
                          "127.0.0.1:" + std::to_string(second.Port())};
    config.httpTimeoutMs = 100;
    auto worker = CreateM800ViewSlaveServer(config);
    auto universe = CreateViewUniverse();
    const auto begin = std::chrono::steady_clock::now();
    const auto status = worker->GetAllViews(*universe);
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    EXPECT_EQ(status.code, StatusCode::TIMEOUT);
    EXPECT_LT(elapsed, std::chrono::milliseconds(180));
    worker->Stop();
}

}  // namespace
}  // namespace kv::test
