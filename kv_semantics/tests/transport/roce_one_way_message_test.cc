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
 * */
#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include "roce/cpu_kv_over_roce_trans_provider.h"
#include "roce/roce_tcp_channel.h"
#include "roce/roce_wire.h"
#include "roce_test_support.h"

namespace kv::test {
namespace {

using Clock = std::chrono::steady_clock;

// Reads one complete frame: the 16 byte header announces the body length.
bool ReadFrame(const roce::TcpChannel& channel, std::vector<std::uint8_t>& frame,
               std::string& error)
{
    std::uint8_t header[roce::kWireHeaderBytes] = {0};
    if (!channel.RecvAll(header, sizeof(header), 4000, error).ok()) { return false; }
    std::uint32_t announced = 0;
    if (!roce::PeekBodyLength(header, sizeof(header), announced)) { return false; }
    const std::size_t bodyLength = announced;
    frame.assign(header, header + sizeof(header));
    if (bodyLength == 0U) { return true; }
    frame.resize(sizeof(header) + bodyLength);
    return channel.RecvAll(frame.data() + sizeof(header), bodyLength, 4000, error).ok();
}

bool SendFrame(const roce::TcpChannel& channel, const std::vector<std::uint8_t>& frame,
               std::string& error)
{
    return channel.SendAll(frame.data(), frame.size(), 4000, error).ok();
}

// A store that follows the link specification but implements only what the
// handshake needs: it answers the capability request and the connect request,
// then reads the connect done and the disconnect announcement without ever
// answering either of them.  A provider that waits for a reply would hang until
// its handshake timeout instead of finishing the call.
void RunSilentStore(std::uint16_t port, std::uint32_t queueNum)
{
    std::string error;
    roce::TcpChannel listener;
    if (!roce::TcpChannel::Listen("127.0.0.1", port, 1, listener, error).ok()) { return; }
    roce::TcpChannel peer;
    if (!roce::TcpChannel::Accept(listener, 5000, peer, error).ok()) { return; }

    std::vector<std::uint8_t> frame;
    if (!ReadFrame(peer, frame, error)) { return; }

    // Table 2: what this store can do.  Only queueNum matters here.
    roce::Capabilities capabilities;
    capabilities.queueNum = static_cast<std::uint16_t>(queueNum);
    capabilities.adminqDepth = 32;
    capabilities.ioqDepth = 256;
    capabilities.ioqKeyBatch = 64;
    capabilities.totalKeyBatch = 128;
    capabilities.srMdts = (1U << 20U);
    capabilities.bsrMdts = (1U << 20U);
    capabilities.storeKeyBatch = 64;
    capabilities.retrieveKeyBatch = 64;
    capabilities.deleteKeyBatch = 16;
    capabilities.existKeyBatch = 64;
    capabilities.keyLength = 8;
    if (!roce::EncodeCapabilityReply(capabilities, frame).ok() || !SendFrame(peer, frame, error)) {
        return;
    }

    if (!ReadFrame(peer, frame, error)) { return; }
    roce::Message request;
    if (!roce::DecodeMessage(frame.data(), frame.size(), request, error).ok()) { return; }

    // Table 3 answer: keep the peer's parameters, hand out queue pair numbers of
    // our own and pick a start PSN.
    roce::QpInfo response = request.qp;
    for (std::uint32_t index = 0; index < response.totalQpNum; ++index) {
        response.qpn[index] = 0x400U + index;
    }
    response.startPsn = 0x8000U;
    if (!roce::EncodeConnectRequest(roce::kWireCmdConnectResponse, response, frame).ok() ||
        !SendFrame(peer, frame, error)) {
        return;
    }

    // Table 4 and table 5 are read and deliberately not answered.
    if (!ReadFrame(peer, frame, error)) { return; }
    static_cast<void>(ReadFrame(peer, frame, error));
}

TEST(CpuKvOverRoceOneWayMessageTest, ConnectDoneIsSentOnceAndNotAwaited)
{
    const std::uint16_t port = ReserveTcpPort();
    ASSERT_NE(0, port);

    std::thread store(RunSilentStore, port, 2U);

    roce::CpuKvOverRoceTransProvider provider(MakeClientConfig(3000));
    std::vector<TransProvider::ConnectionHandle> handles;
    const Clock::time_point started = Clock::now();
    const Status status = ConnectWithRetry(provider, port, 2U, handles);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started).count();
    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(2U, handles.size());

    // The store never answers the connect done message.  Returning before the
    // handshake timeout shows the message is treated as one way.
    EXPECT_LT(elapsed, 2500) << "the connect done message appears to be awaited";

    const std::vector<Status> deleted = provider.DeleteConnections(handles);
    for (const Status& result : deleted) { EXPECT_TRUE(result.ok()) << result.message; }
    EXPECT_EQ(0U, provider.ConnectionCount());
    store.join();
}

TEST(CpuKvOverRoceOneWayMessageTest, DisconnectReturnsWithoutWaitingForAnAnswer)
{
    const std::uint16_t port = ReserveTcpPort();
    ASSERT_NE(0, port);

    std::thread store(RunSilentStore, port, 1U);

    roce::CpuKvOverRoceTransProvider provider(MakeClientConfig(3000));
    std::vector<TransProvider::ConnectionHandle> handles;
    const Status status = ConnectWithRetry(provider, port, 1U, handles);
    ASSERT_TRUE(status.ok()) << status.message;

    const Clock::time_point started = Clock::now();
    const std::vector<Status> deleted = provider.DeleteConnections(handles);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started).count();
    ASSERT_EQ(1U, deleted.size());
    EXPECT_TRUE(deleted.front().ok()) << deleted.front().message;
    EXPECT_LT(elapsed, 500) << "the disconnect request appears to be awaited";
    EXPECT_EQ(0U, provider.ConnectionCount());
    EXPECT_EQ(0U, provider.HandleCount());
    store.join();
}

}  // namespace
}  // namespace kv::test
