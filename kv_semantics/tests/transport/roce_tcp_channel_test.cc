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

#include "roce/roce_tcp_channel.h"
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include "roce_test_support.h"

namespace kv::test {
namespace {

TEST(RoceTcpChannelTest, ConnectsWithOneMegabyteBuffersAndFiveSecondUserTimeout)
{
    const std::uint16_t port = ReserveTcpPort();
    ASSERT_NE(0, port);

    roce::TcpChannel listener;
    std::string error;
    ASSERT_TRUE(roce::TcpChannel::Listen("127.0.0.1", port, 1, listener, error).ok()) << error;

    roce::TcpChannel accepted;
    Status acceptStatus = Status::Error(StatusCode::INTERNAL_ERROR, "not executed");
    std::thread acceptThread(
        [&]() { acceptStatus = roce::TcpChannel::Accept(listener, 4000, accepted, error); });

    roce::TcpChannel client;
    Status connectStatus = Status::Error(StatusCode::INTERNAL_ERROR, "not executed");
    for (int attempt = 0; attempt < 100 && !connectStatus.ok(); ++attempt) {
        connectStatus =
            roce::TcpChannel::Connect("127.0.0.1", "127.0.0.1", port, 2000, client, error);
        if (!connectStatus.ok()) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
    }
    acceptThread.join();

    ASSERT_TRUE(connectStatus.ok()) << connectStatus.message;
    ASSERT_TRUE(acceptStatus.ok()) << acceptStatus.message;

    // The kernel doubles the requested buffer before reporting it back, and it
    // clamps to net.core.{r,w}mem_max, so only the lower bound is asserted.
    EXPECT_GE(client.SendBufferBytes(), roce::TcpChannel::kSocketBufferBytes);
    EXPECT_GE(client.ReceiveBufferBytes(), roce::TcpChannel::kSocketBufferBytes);
    EXPECT_GE(accepted.SendBufferBytes(), roce::TcpChannel::kSocketBufferBytes);
    EXPECT_GE(accepted.ReceiveBufferBytes(), roce::TcpChannel::kSocketBufferBytes);

    EXPECT_EQ(roce::TcpChannel::kUserTimeoutMs, client.UserTimeoutMs());
    EXPECT_EQ(roce::TcpChannel::kUserTimeoutMs, accepted.UserTimeoutMs());

    EXPECT_FALSE(client.PeerAddress().empty());
    EXPECT_FALSE(accepted.PeerAddress().empty());
}

TEST(RoceTcpChannelTest, ALongTimeoutNeitherFailsImmediatelyNorBlocksForever)
{
    const std::uint16_t port = ReserveTcpPort();
    ASSERT_NE(0, port);

    roce::TcpChannel listener;
    std::string error;
    ASSERT_TRUE(roce::TcpChannel::Listen("127.0.0.1", port, 1, listener, error).ok()) << error;

    roce::TcpChannel accepted;
    Status acceptStatus = Status::Error(StatusCode::INTERNAL_ERROR, "not executed");
    std::thread acceptThread(
        [&]() { acceptStatus = roce::TcpChannel::Accept(listener, 4000, accepted, error); });

    roce::TcpChannel client;
    Status connectStatus = Status::Error(StatusCode::INTERNAL_ERROR, "not executed");
    for (int attempt = 0; attempt < 100 && !connectStatus.ok(); ++attempt) {
        connectStatus =
            roce::TcpChannel::Connect("127.0.0.1", "127.0.0.1", port, 2000, client, error);
        if (!connectStatus.ok()) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
    }
    acceptThread.join();
    ASSERT_TRUE(connectStatus.ok()) << connectStatus.message;
    ASSERT_TRUE(acceptStatus.ok()) << acceptStatus.message;

    // UINT32_MAX milliseconds has to mean "practically forever": neither a poll
    // timeout that wrapped negative (endless wait) nor a deadline that overflowed
    // (immediate timeout).  A peer that closes after 150 ms ends the wait.
    std::atomic<bool> finished{false};
    Status receiveStatus = Status::Error(StatusCode::INTERNAL_ERROR, "not executed");
    std::uint8_t byte = 0;
    std::thread receiver([&]() {
        receiveStatus = client.RecvAll(&byte, 1, 0xFFFFFFFFU, error);
        finished = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    EXPECT_FALSE(finished.load());
    accepted = roce::TcpChannel();
    receiver.join();
    EXPECT_FALSE(receiveStatus.ok());
}

TEST(RoceTcpChannelTest, ClosedChannelReportsNoTuning)
{
    roce::TcpChannel channel;
    EXPECT_FALSE(channel.Valid());
    EXPECT_EQ(0U, channel.SendBufferBytes());
    EXPECT_EQ(0U, channel.ReceiveBufferBytes());
    EXPECT_EQ(0U, channel.UserTimeoutMs());
}

}  // namespace
}  // namespace kv::test
