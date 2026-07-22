#pragma once

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include "core/transport.h"
#include "two_sided/tcp/tcp_message_channel.h"

namespace transport::test {

inline const char* statusName(Status status)
{
    if (status.Success()) { return "Ok"; }
    return status == Status::InvalidParam() ? "InvalidArgument" : "Failed";
}

inline bool expectOk(Status status, const char* step)
{
    if (status == Status::OK()) { return true; }
    std::cerr << step << " failed: " << statusName(status) << '\n';
    return false;
}

inline bool expectStatus(Status actual, Status expected, const char* step)
{
    if (actual == expected) { return true; }
    std::cerr << step << " failed: got " << statusName(actual) << ", expected "
              << statusName(expected) << '\n';
    return false;
}

inline bool expectTrue(bool value, const char* step)
{
    if (value) { return true; }
    std::cerr << step << " failed\n";
    return false;
}

inline bool envEnabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && std::string(value) == "1";
}

inline uint16_t envPort(const char* name, uint16_t fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') { return fallback; }
    const auto value = std::strtoul(text, nullptr, 10);
    if (value == 0 || value > UINT16_MAX) { return fallback; }
    return static_cast<uint16_t>(value);
}

inline Endpoint makeEndpoint(const std::string& host, uint16_t port)
{
    Endpoint result;
    result.host = host;
    result.port = port;
    return result;
}

inline bool sendTextWithRetry(TcpMessageChannel& tcp, const Endpoint& peer, const std::string& text,
                              int attempts, int interval_ms, const char* step)
{
    for (int attempt = 1; attempt <= attempts; ++attempt) {
        if (tcp.Send(peer, text.data(), text.size()) == Status::OK()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
    std::cerr << step << " failed\n";
    return false;
}

inline bool receiveText(TcpMessageChannel& tcp, Endpoint& peer, std::string& text, const char* step)
{
    Metadata data;
    const auto status = tcp.Receive(peer, data);
    if (status != Status::OK()) {
        std::cerr << step << " failed: " << statusName(status) << '\n';
        return false;
    }
    text.assign(data.begin(), data.end());
    return true;
}

inline bool receiveText(TcpMessageChannel& tcp, std::string& text, const char* step)
{
    Endpoint peer;
    return receiveText(tcp, peer, text, step);
}

}  // namespace transport::test
