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
#include "ipc/uds.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#include "status/status.h"

namespace UC::Transfer {
namespace {

std::string MakeSocketPath()
{
    static std::atomic<std::uint64_t> counter{0};
    return "/tmp/ucm_uds_test_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter.fetch_add(1)) + ".sock";
}

class SocketPathGuard {
public:
    SocketPathGuard() : path_{MakeSocketPath()} {}
    ~SocketPathGuard() { ::unlink(path_.c_str()); }
    SocketPathGuard(const SocketPathGuard&) = delete;
    SocketPathGuard& operator=(const SocketPathGuard&) = delete;
    const std::string& Get() const { return path_; }

private:
    std::string path_;
};

std::vector<std::byte> MakePattern(std::size_t size, std::uint8_t seed)
{
    std::vector<std::byte> bytes(size);
    for (std::size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<std::byte>(seed + static_cast<std::uint8_t>(i));
    }
    return bytes;
}

/** Test fixture owning one listener plus one accepted client/server pair. */
class UdsPairTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_TRUE(listener_.Listen(path_.Get()).Success());
        auto client = UdsConnection::Connect(path_.Get());
        ASSERT_TRUE(client.HasValue());
        auto server = listener_.Accept();
        ASSERT_TRUE(server.HasValue());
        client_ = std::move(client.Value());
        server_ = std::move(server.Value());
    }

    SocketPathGuard path_;
    UdsListener listener_;
    UdsConnection client_;
    UdsConnection server_;
    std::vector<std::byte> payload_;
    std::vector<int> fds_;
};

TEST_F(UdsPairTest, FrameRoundTripBothDirections)
{
    const std::vector<std::byte> outgoing = MakePattern(64, 1);
    ASSERT_TRUE(client_.SendFrame(outgoing).Success());
    ASSERT_TRUE(server_.ReceiveFrame(payload_, fds_).Success());
    EXPECT_EQ(payload_, outgoing);
    EXPECT_TRUE(fds_.empty());

    const std::vector<std::byte> incoming = MakePattern(128, 9);
    ASSERT_TRUE(server_.SendFrame(incoming).Success());
    ASSERT_TRUE(client_.ReceiveFrame(payload_, fds_).Success());
    EXPECT_EQ(payload_, incoming);
    EXPECT_TRUE(fds_.empty());
}

TEST_F(UdsPairTest, EmptyPayloadFrameRoundTrip)
{
    const std::vector<std::byte> empty;
    ASSERT_TRUE(client_.SendFrame(empty).Success());
    ASSERT_TRUE(server_.ReceiveFrame(payload_, fds_).Success());
    EXPECT_TRUE(payload_.empty());
    EXPECT_TRUE(fds_.empty());

    ASSERT_TRUE(server_.SendFrame(empty).Success());
    ASSERT_TRUE(client_.ReceiveFrame(payload_, fds_).Success());
    EXPECT_TRUE(payload_.empty());
    EXPECT_TRUE(fds_.empty());
}

TEST_F(UdsPairTest, LargeFrameRoundTrip)
{
    const std::vector<std::byte> big = MakePattern(1u << 20, 0xAB);
    Status sendStatus = Status::Ok();
    std::thread sender([this, &big, &sendStatus] { sendStatus = client_.SendFrame(big); });
    ASSERT_TRUE(server_.ReceiveFrame(payload_, fds_).Success());
    sender.join();
    ASSERT_TRUE(sendStatus.Success());
    EXPECT_EQ(payload_.size(), big.size());
    EXPECT_EQ(payload_, big);
}

TEST_F(UdsPairTest, SequentialFramesKeepBoundaries)
{
    const std::vector<std::byte> first = MakePattern(1, 1);
    const std::vector<std::byte> second = MakePattern(1000, 2);
    const std::vector<std::byte> third = MakePattern(5, 3);
    ASSERT_TRUE(client_.SendFrame(first).Success());
    ASSERT_TRUE(client_.SendFrame(second).Success());
    ASSERT_TRUE(client_.SendFrame(third).Success());

    ASSERT_TRUE(server_.ReceiveFrame(payload_, fds_).Success());
    EXPECT_EQ(payload_, first);
    ASSERT_TRUE(server_.ReceiveFrame(payload_, fds_).Success());
    EXPECT_EQ(payload_, second);
    ASSERT_TRUE(server_.ReceiveFrame(payload_, fds_).Success());
    EXPECT_EQ(payload_, third);
    EXPECT_TRUE(fds_.empty());
}

TEST_F(UdsPairTest, ScmRightsDeliversWorkingFd)
{
    std::array<int, 2> channel{};
    ASSERT_EQ(::pipe(channel.data()), 0);
    const std::vector<std::byte> message = MakePattern(16, 7);
    const Status sendStatus = client_.SendFrameWithFds(message, {channel[1]});
    ASSERT_TRUE(sendStatus.Success()) << sendStatus.ToString();
    const Status recvStatus = server_.ReceiveFrame(payload_, fds_);
    ASSERT_TRUE(recvStatus.Success()) << recvStatus.ToString();
    ASSERT_EQ(fds_.size(), 1u);
    EXPECT_EQ(payload_, message);

    ASSERT_EQ(::write(fds_[0], "x", 1), 1);  // server writes through the received duplicate
    char observed = 0;
    ASSERT_EQ(::read(channel[0], &observed, 1), 1);  // client reads it from its own read end
    EXPECT_EQ(observed, 'x');

    ::close(fds_[0]);
    ::close(channel[0]);
    ::close(channel[1]);
}

TEST_F(UdsPairTest, ScmRightsDeliversMultipleFds)
{
    std::array<int, 2> channelA{};
    std::array<int, 2> channelB{};
    ASSERT_EQ(::pipe(channelA.data()), 0);
    ASSERT_EQ(::pipe(channelB.data()), 0);
    const std::vector<std::byte> message = MakePattern(8, 4);
    ASSERT_TRUE(client_.SendFrameWithFds(message, {channelA[1], channelB[1]}).Success());
    ASSERT_TRUE(server_.ReceiveFrame(payload_, fds_).Success());
    ASSERT_EQ(fds_.size(), 2u);
    EXPECT_EQ(payload_, message);

    ASSERT_EQ(::write(fds_[0], "a", 1), 1);
    ASSERT_EQ(::write(fds_[1], "b", 1), 1);
    char observedA = 0;
    char observedB = 0;
    ASSERT_EQ(::read(channelA[0], &observedA, 1), 1);
    ASSERT_EQ(::read(channelB[0], &observedB, 1), 1);
    EXPECT_EQ(observedA, 'a');
    EXPECT_EQ(observedB, 'b');

    ::close(fds_[0]);
    ::close(fds_[1]);
    ::close(channelA[0]);
    ::close(channelA[1]);
    ::close(channelB[0]);
    ::close(channelB[1]);
}

TEST_F(UdsPairTest, SendRejectsTooManyFds)
{
    const std::vector<int> tooMany(kUdsMaxFdsPerFrame + 1, 0);
    EXPECT_EQ(client_.SendFrameWithFds(MakePattern(4, 1), tooMany), Status::InvalidParam());
}

TEST_F(UdsPairTest, ReceiveYieldsErrorAfterPeerClose)
{
    client_.Close();
    const Status st = server_.ReceiveFrame(payload_, fds_);
    EXPECT_FALSE(st.Success());
    EXPECT_NE(st, Status::InvalidParam());  // peer EOF is an I/O error, not a usage error
}

TEST_F(UdsPairTest, SendYieldsErrorAfterPeerClose)
{
    client_.Close();
    (void)server_.ReceiveFrame(payload_, fds_);  // observe the EOF, letting the stack settle
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    const std::vector<std::byte> message = MakePattern(32, 3);
    Status st = server_.SendFrame(message);
    if (st.Success()) { st = server_.SendFrame(message); }  // EPIPE follows the RST
    EXPECT_FALSE(st.Success());
}

TEST_F(UdsPairTest, SendRejectsOversizedPayload)
{
    const std::vector<std::byte> huge(kUdsMaxFrameSize + 1);
    EXPECT_EQ(client_.SendFrame(huge), Status::InvalidParam());
}

TEST_F(UdsPairTest, ClosedConnectionRejectsOperations)
{
    client_.Close();
    EXPECT_FALSE(client_.IsValid());
    EXPECT_EQ(client_.SendFrame(MakePattern(4, 1)), Status::InvalidParam());
    EXPECT_EQ(client_.ReceiveFrame(payload_, fds_), Status::InvalidParam());
}

TEST_F(UdsPairTest, MovePreservesEndpoint)
{
    UdsConnection moved = std::move(client_);
    EXPECT_FALSE(client_.IsValid());
    EXPECT_TRUE(moved.IsValid());
    const std::vector<std::byte> message = MakePattern(4, 9);
    ASSERT_TRUE(moved.SendFrame(message).Success());
    ASSERT_TRUE(server_.ReceiveFrame(payload_, fds_).Success());
    EXPECT_EQ(payload_, message);
    EXPECT_TRUE(server_.IsValid());
}

TEST(UdsListenerTest, ListenFailsOnOccupiedPath)
{
    SocketPathGuard path;
    UdsListener first;
    ASSERT_TRUE(first.Listen(path.Get()).Success());
    UdsListener second;
    const Status st = second.Listen(path.Get());
    EXPECT_FALSE(st.Success());
    EXPECT_EQ(st, Status::OsApiError());
}

TEST(UdsListenerTest, ListenRejectsEmptyPath)
{
    UdsListener listener;
    EXPECT_EQ(listener.Listen(""), Status::InvalidParam());
}

TEST(UdsListenerTest, MovePreservesListener)
{
    SocketPathGuard path;
    UdsListener listener;
    ASSERT_TRUE(listener.Listen(path.Get()).Success());
    UdsListener moved = std::move(listener);
    EXPECT_FALSE(listener.IsValid());
    EXPECT_TRUE(moved.IsValid());
    EXPECT_EQ(moved.Path(), path.Get());
    auto client = UdsConnection::Connect(path.Get());
    ASSERT_TRUE(client.HasValue());
    auto server = moved.Accept();
    ASSERT_TRUE(server.HasValue());
    EXPECT_TRUE(server.Value().IsValid());
}

TEST(UdsConnectionTest, ConnectFailsOnMissingPath)
{
    auto res = UdsConnection::Connect("/tmp/ucm_uds_test_no_such_socket.sock");
    ASSERT_FALSE(res.HasValue());
    EXPECT_EQ(res.Error(), Status::OsApiError());
}

TEST(UdsConnectionTest, ConnectRejectsEmptyPath)
{
    auto res = UdsConnection::Connect("");
    ASSERT_FALSE(res.HasValue());
    EXPECT_EQ(res.Error(), Status::InvalidParam());
}

TEST(UdsFrameTest, ReceiveRejectsOversizedHeader)
{
    SocketPathGuard path;
    UdsListener listener;
    ASSERT_TRUE(listener.Listen(path.Get()).Success());

    const int raw = ::socket(AF_UNIX, SOCK_STREAM, 0);
    ASSERT_GE(raw, 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.Get().c_str(), sizeof(addr.sun_path) - 1);
    ASSERT_EQ(::connect(raw, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    auto serverRes = listener.Accept();
    ASSERT_TRUE(serverRes.HasValue());
    UdsConnection server = std::move(serverRes.Value());

    const std::uint32_t badLength = kUdsMaxFrameSize + 1;
    const std::array<std::uint8_t, 4> head{static_cast<std::uint8_t>(badLength & 0xFF),
                                           static_cast<std::uint8_t>((badLength >> 8) & 0xFF),
                                           static_cast<std::uint8_t>((badLength >> 16) & 0xFF),
                                           static_cast<std::uint8_t>((badLength >> 24) & 0xFF)};
    ASSERT_EQ(::send(raw, head.data(), head.size(), 0), static_cast<ssize_t>(head.size()));

    std::vector<std::byte> payload;
    std::vector<int> fds;
    EXPECT_EQ(server.ReceiveFrame(payload, fds), Status::InvalidParam());
    ::close(raw);
}

}  // namespace
}  // namespace UC::Transfer
