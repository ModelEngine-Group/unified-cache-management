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
#include "link_protocol.h"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>

namespace UC::ASU {
namespace {

class LinkProtoPackTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LinkProtoPackTest, NegotiateSqePackMatchesProtocol)
{
    NegotiateRequest req;
    req.cap = 0;
    req.private_len = 4;
    req.major_version = 1;
    req.minor_version = 0;
    req.kato = 30;

    NegotiateSqe sqe;
    auto status = sqe.Pack(req);
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint8_t> expected(kMsgHeaderSize + kNegotiatePayloadSize, 0);

    // Header: crc(4)=0, ver(1)=1, cmd(1)=0, pad(6)=0, len(4)=160
    expected[4] = 1;  // ver
    expected[5] = 0;  // cmd = Negotiate
    std::uint32_t payload_len = kNegotiatePayloadSize;
    std::memcpy(&expected[12], &payload_len, 4);

    // Offset 16: cap[31:0] = 0
    // Offset 20: rsv[23:0] = 0 (already zero)
    // Offset 44: private_len[31:0]
    std::memcpy(&expected[44], &req.private_len, 4);
    // Offset 48: major_version
    expected[48] = req.major_version;
    // Offset 49: minor_version
    expected[49] = req.minor_version;
    // Offset 50: kato[15:0]
    std::memcpy(&expected[50], &req.kato, 2);

    ASSERT_EQ(sqe.Size(), expected.size());
    const auto* packed = static_cast<const std::uint8_t*>(sqe.Data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i])
            << "Mismatch at byte " << i << ": expected 0x" << std::hex
            << static_cast<int>(expected[i]) << ", got 0x" << static_cast<int>(packed[i]);
    }
}

TEST_F(LinkProtoPackTest, HandshakeSqePackMatchesProtocol)
{
    HandshakeRequest req;
    for (int i = 0; i < 16; ++i) { req.gid[i] = static_cast<std::uint8_t>(0xA0 + i); }
    req.lid = 0x1234;
    req.mtu = 4;  // 2048 bytes
    req.total_qp_num = 8;
    req.sl = 1;
    req.traffic_class = 2;
    req.rnr_timer = 3;
    req.rnr_retry_cnt = 7;
    req.timeout = 14;
    req.retry_cnt = 5;
    req.qp_rd_atom = 6;
    req.rsv = 0;
    req.start_psn = 0xDEADBEEF;
    for (int i = 0; i < 32; ++i) { req.qpn[i] = 0x1000 + i; }

    HandshakeSqe sqe;
    auto status = sqe.Pack(req);
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint8_t> expected(kMsgHeaderSize + kHandshakePayloadSize, 0);

    // Header: crc(4)=0, ver(1)=1, cmd(1)=1, pad(6)=0, len(4)=160
    expected[4] = 1;  // ver
    expected[5] = 1;  // cmd = Handshake
    std::uint32_t payload_len = kHandshakePayloadSize;
    std::memcpy(&expected[12], &payload_len, 4);

    // Offset 16: gid[15:0]
    std::memcpy(&expected[16], req.gid, 16);
    // Offset 32: lid[15:0]
    std::memcpy(&expected[32], &req.lid, 2);
    // Offset 34: mtu
    expected[34] = req.mtu;
    // Offset 35: total_qp_num
    expected[35] = req.total_qp_num;
    // Offset 36: sl
    expected[36] = req.sl;
    // Offset 37: traffic_class
    expected[37] = req.traffic_class;
    // Offset 38: rnr_timer
    expected[38] = req.rnr_timer;
    // Offset 39: rnr_retry_cnt
    expected[39] = req.rnr_retry_cnt;
    // Offset 40: timeout
    expected[40] = req.timeout;
    // Offset 41: retry_cnt
    expected[41] = req.retry_cnt;
    // Offset 42: qp_rd_atom
    expected[42] = req.qp_rd_atom;
    // Offset 43: rsv
    expected[43] = req.rsv;
    // Offset 44: start_psn[31:0]
    std::memcpy(&expected[44], &req.start_psn, 4);
    // Offset 48: qpn[32] (128 bytes)
    std::memcpy(&expected[48], req.qpn, 128);

    ASSERT_EQ(sqe.Size(), expected.size());
    const auto* packed = static_cast<const std::uint8_t*>(sqe.Data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i])
            << "Mismatch at byte " << i << ": expected 0x" << std::hex
            << static_cast<int>(expected[i]) << ", got 0x" << static_cast<int>(packed[i]);
    }
}

TEST_F(LinkProtoPackTest, HandshakeDoneSqePackMatchesProtocol)
{
    HandshakeDoneSqe sqe;
    auto status = sqe.Pack();
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint8_t> expected(kMsgHeaderSize, 0);

    // Header: crc(4)=0, ver(1)=1, cmd(1)=3, pad(6)=0, len(4)=0
    expected[4] = 1;  // ver
    expected[5] = 3;  // cmd = HandshakeDone
    std::uint32_t payload_len = 0;
    std::memcpy(&expected[12], &payload_len, 4);

    ASSERT_EQ(sqe.Size(), expected.size());
    const auto* packed = static_cast<const std::uint8_t*>(sqe.Data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i])
            << "Mismatch at byte " << i << ": expected 0x" << std::hex
            << static_cast<int>(expected[i]) << ", got 0x" << static_cast<int>(packed[i]);
    }
}

TEST_F(LinkProtoPackTest, DisconnectSqePackMatchesProtocol)
{
    DisconnectRequest req;
    req.local_qpn = 0x00010001;
    req.remote_qpn = 0x00020002;

    DisconnectSqe sqe;
    auto status = sqe.Pack(req);
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint8_t> expected(kMsgHeaderSize + kDisconnectPayloadSize, 0);

    // Header: crc(4)=0, ver(1)=1, cmd(1)=4, pad(6)=0, len(4)=8
    expected[4] = 1;  // ver
    expected[5] = 4;  // cmd = Disconnect
    std::uint32_t payload_len = kDisconnectPayloadSize;
    std::memcpy(&expected[12], &payload_len, 4);

    // Offset 16: local_qpn[31:0]
    std::memcpy(&expected[16], &req.local_qpn, 4);
    // Offset 20: remote_qpn[31:0]
    std::memcpy(&expected[20], &req.remote_qpn, 4);

    ASSERT_EQ(sqe.Size(), expected.size());
    const auto* packed = static_cast<const std::uint8_t*>(sqe.Data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i])
            << "Mismatch at byte " << i << ": expected 0x" << std::hex
            << static_cast<int>(expected[i]) << ", got 0x" << static_cast<int>(packed[i]);
    }
}

}  // namespace
}  // namespace UC::ASU
