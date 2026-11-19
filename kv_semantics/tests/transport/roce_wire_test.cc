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

#include "roce/roce_wire.h"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace kv::test {
namespace {

using roce::Capabilities;
using roce::CapabilityRequest;
using roce::Message;
using roce::QpInfo;

std::uint32_t Crc32(const std::uint8_t* data, std::size_t size)
{
    // Same construction as the wire code: reflected CRC-32C with a zero
    // initial value and no final inversion.
    std::uint32_t crc = 0;
    for (std::size_t index = 0; index < size; ++index) {
        crc ^= data[index];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 1U) != 0U ? (0x82F63B78U ^ (crc >> 1U)) : (crc >> 1U);
        }
    }
    return crc;
}

std::uint32_t ReadU32(const std::vector<std::uint8_t>& frame, std::size_t offset)
{
    return (static_cast<std::uint32_t>(frame[offset]) << 24U) |
           (static_cast<std::uint32_t>(frame[offset + 1]) << 16U) |
           (static_cast<std::uint32_t>(frame[offset + 2]) << 8U) |
           static_cast<std::uint32_t>(frame[offset + 3]);
}

void WriteU32(std::vector<std::uint8_t>& frame, std::size_t offset, std::uint32_t value)
{
    frame[offset] = static_cast<std::uint8_t>(value >> 24U);
    frame[offset + 1] = static_cast<std::uint8_t>(value >> 16U);
    frame[offset + 2] = static_cast<std::uint8_t>(value >> 8U);
    frame[offset + 3] = static_cast<std::uint8_t>(value);
}

// The private area of the capability messages is little endian.
void WriteU16Le(std::vector<std::uint8_t>& frame, std::size_t offset, std::uint16_t value)
{
    frame[offset] = static_cast<std::uint8_t>(value);
    frame[offset + 1] = static_cast<std::uint8_t>(value >> 8U);
}

void WriteU32Le(std::vector<std::uint8_t>& frame, std::size_t offset, std::uint32_t value)
{
    frame[offset] = static_cast<std::uint8_t>(value);
    frame[offset + 1] = static_cast<std::uint8_t>(value >> 8U);
    frame[offset + 2] = static_cast<std::uint8_t>(value >> 16U);
    frame[offset + 3] = static_cast<std::uint8_t>(value >> 24U);
}

// Builds the capability reply as an older peer that predates
// delete_key_batch would: same fields, private_len 36.
std::vector<std::uint8_t> LegacyCapabilityReply(const Capabilities& capabilities)
{
    std::vector<std::uint8_t> frame(roce::kWireHeaderBytes + roce::kWireBodyBytes, 0U);
    frame[4] = roce::kWireVersion;
    frame[5] = roce::kWireCmdCapability;
    WriteU32(frame, 12, static_cast<std::uint32_t>(roce::kWireBodyBytes));
    std::size_t offset = roce::kWireHeaderBytes + 28;
    WriteU32(frame, offset, 36);
    offset += 4;
    frame[offset++] = capabilities.majorVersion;
    frame[offset++] = capabilities.minorVersion;
    WriteU16Le(frame, offset, capabilities.queueNum);
    offset += 2;
    WriteU16Le(frame, offset, capabilities.adminqDepth);
    offset += 2;
    WriteU16Le(frame, offset, capabilities.ioqDepth);
    offset += 2;
    WriteU16Le(frame, offset, capabilities.ioqKeyBatch);
    offset += 2;
    WriteU16Le(frame, offset, capabilities.totalKeyBatch);
    offset += 2;
    WriteU32Le(frame, offset, capabilities.srMdts);
    offset += 4;
    WriteU32Le(frame, offset, capabilities.bsrMdts);
    offset += 4;
    WriteU16Le(frame, offset, capabilities.storeKeyBatch);
    offset += 2;
    WriteU16Le(frame, offset, capabilities.retrieveKeyBatch);
    offset += 2;
    WriteU16Le(frame, offset, capabilities.existKeyBatch);
    offset += 2;
    frame[offset++] = capabilities.keyLength;
    frame[offset++] = 0;  // pad
    WriteU16Le(frame, offset, capabilities.controllerId);
    offset += 2;
    WriteU32Le(frame, offset, capabilities.controllerCap);
    WriteU32(frame, 0, Crc32(frame.data() + 4, frame.size() - 4));
    return frame;
}

Capabilities SampleCapabilities()
{
    Capabilities capabilities;
    capabilities.majorVersion = 1;
    capabilities.minorVersion = 2;
    capabilities.queueNum = 4;
    capabilities.adminqDepth = 8;
    capabilities.ioqDepth = 512;
    capabilities.ioqKeyBatch = 16;
    capabilities.totalKeyBatch = 1024;
    capabilities.srMdts = 1U << 20U;
    capabilities.bsrMdts = 1U << 19U;
    capabilities.storeKeyBatch = 8;
    capabilities.retrieveKeyBatch = 8;
    capabilities.deleteKeyBatch = 16;
    capabilities.existKeyBatch = 32;
    capabilities.keyLength = 8;
    capabilities.controllerId = 3;
    capabilities.controllerCap = 0xdeadbeefU;
    return capabilities;
}

QpInfo SampleQpInfo()
{
    QpInfo info;
    for (std::size_t index = 0; index < sizeof(info.gid); ++index) {
        info.gid[index] = static_cast<std::uint8_t>(index + 1);
    }
    info.lid = 0x1234;
    info.mtu = 5;
    info.totalQpNum = 2;
    info.sl = 3;
    info.trafficClass = 104;
    info.startPsn = 0x1000;
    info.qpn[0] = 101;
    info.qpn[1] = 202;
    return info;
}

TEST(RoceWireTest, ChecksumIsCrc32cStartingFromZero)
{
    // Reflected CRC-32C with a zero initial value and no final inversion.
    // The standard CRC-32C of this vector is 0xE3069283 (init and xorout both
    // all ones); zeroing both leaves 0x58E3FA20.
    const std::string check = "123456789";
    EXPECT_EQ(0x58E3FA20U,
              Crc32(reinterpret_cast<const std::uint8_t*>(check.data()), check.size()));
    EXPECT_EQ(0U, Crc32(nullptr, 0));
}

TEST(RoceWireTest, CapabilityRequestPutsTheProtocolVersionAfterPrivateLength)
{
    std::vector<std::uint8_t> frame;
    ASSERT_TRUE(roce::EncodeCapabilityRequest(CapabilityRequest{}, frame).ok());

    // table 1: header 16 + body 160, crc/(ver|cmd)/len in the documented places.
    ASSERT_EQ(roce::kWireHeaderBytes + roce::kWireBodyBytes, frame.size());
    EXPECT_EQ(roce::kWireVersion, frame[4]);
    EXPECT_EQ(roce::kWireCmdCapability, frame[5]);
    EXPECT_EQ(roce::kWireBodyBytes, ReadU32(frame, 12));

    // private_len sits at body offset 28 and the request payload is 4 bytes.
    EXPECT_EQ(4U, ReadU32(frame, roce::kWireHeaderBytes + 28));
    EXPECT_EQ(1U, frame[roce::kWireHeaderBytes + 32]);  // major
    EXPECT_EQ(0U, frame[roce::kWireHeaderBytes + 33]);  // minor
    EXPECT_EQ(0U, frame[roce::kWireHeaderBytes + 34]);  // kato low byte
    EXPECT_EQ(0U, frame[roce::kWireHeaderBytes + 35]);  // kato high byte
}

TEST(RoceWireTest, CapabilityReplyRoundTripsEveryField)
{
    std::vector<std::uint8_t> frame;
    const Capabilities expected = SampleCapabilities();
    ASSERT_TRUE(roce::EncodeCapabilityReply(expected, frame).ok());

    Message message;
    std::string error;
    ASSERT_TRUE(roce::DecodeMessage(frame.data(), frame.size(), message, error).ok()) << error;

    EXPECT_EQ(roce::kWireCmdCapability, message.cmd);
    EXPECT_TRUE(message.capabilityReply);
    EXPECT_EQ(expected.majorVersion, message.capabilities.majorVersion);
    EXPECT_EQ(expected.minorVersion, message.capabilities.minorVersion);
    EXPECT_EQ(expected.queueNum, message.capabilities.queueNum);
    EXPECT_EQ(expected.adminqDepth, message.capabilities.adminqDepth);
    EXPECT_EQ(expected.ioqDepth, message.capabilities.ioqDepth);
    EXPECT_EQ(expected.ioqKeyBatch, message.capabilities.ioqKeyBatch);
    EXPECT_EQ(expected.totalKeyBatch, message.capabilities.totalKeyBatch);
    EXPECT_EQ(expected.srMdts, message.capabilities.srMdts);
    EXPECT_EQ(expected.bsrMdts, message.capabilities.bsrMdts);
    EXPECT_EQ(expected.storeKeyBatch, message.capabilities.storeKeyBatch);
    EXPECT_EQ(expected.retrieveKeyBatch, message.capabilities.retrieveKeyBatch);
    EXPECT_EQ(expected.deleteKeyBatch, message.capabilities.deleteKeyBatch);
    EXPECT_EQ(expected.existKeyBatch, message.capabilities.existKeyBatch);
    EXPECT_EQ(expected.keyLength, message.capabilities.keyLength);
    EXPECT_EQ(expected.controllerId, message.capabilities.controllerId);
    EXPECT_EQ(expected.controllerCap, message.capabilities.controllerCap);
}

TEST(RoceWireTest, CapabilityPrivateAreaIsTheOnlyLittleEndianPart)
{
    std::vector<std::uint8_t> frame;
    ASSERT_TRUE(roce::EncodeCapabilityReply(SampleCapabilities(), frame).ok());

    // The header length stays big endian: 160 is 0x000000A0.
    EXPECT_EQ(0U, frame[12]);
    EXPECT_EQ(0U, frame[13]);
    EXPECT_EQ(0U, frame[14]);
    EXPECT_EQ(160U, frame[15]);
    // queue_num = 4 sits at private offset 2 and travels little endian.
    EXPECT_EQ(4U, frame[roce::kWireHeaderBytes + 32U + 2U]);
    EXPECT_EQ(0U, frame[roce::kWireHeaderBytes + 32U + 3U]);
    // private_len itself is outside the private area and stays big endian.
    EXPECT_EQ(38U, ReadU32(frame, roce::kWireHeaderBytes + 28U));
}

TEST(RoceWireTest, ConnectRequestRoundTripsTheQueuePairTable)
{
    std::vector<std::uint8_t> frame;
    const QpInfo expected = SampleQpInfo();
    ASSERT_TRUE(roce::EncodeConnectRequest(roce::kWireCmdConnectRequest, expected, frame).ok());
    // header 16 + 32 fixed bytes + one 32 bit qpn per queue pair
    ASSERT_EQ(roce::kWireHeaderBytes + 32U + 4U * expected.totalQpNum, frame.size());
    EXPECT_EQ(roce::kWireCmdConnectRequest, frame[5]);

    Message message;
    std::string error;
    ASSERT_TRUE(roce::DecodeMessage(frame.data(), frame.size(), message, error).ok()) << error;

    EXPECT_EQ(roce::kWireCmdConnectRequest, message.cmd);
    EXPECT_EQ(0, std::memcmp(expected.gid, message.qp.gid, sizeof(expected.gid)));
    EXPECT_EQ(expected.lid, message.qp.lid);
    EXPECT_EQ(expected.mtu, message.qp.mtu);
    EXPECT_EQ(expected.totalQpNum, message.qp.totalQpNum);
    EXPECT_EQ(expected.sl, message.qp.sl);
    EXPECT_EQ(expected.trafficClass, message.qp.trafficClass);
    EXPECT_EQ(expected.rnrTimer, message.qp.rnrTimer);
    EXPECT_EQ(expected.rnrRetryCnt, message.qp.rnrRetryCnt);
    EXPECT_EQ(expected.timeout, message.qp.timeout);
    EXPECT_EQ(expected.retryCnt, message.qp.retryCnt);
    EXPECT_EQ(expected.qpRdAtom, message.qp.qpRdAtom);
    EXPECT_EQ(expected.startPsn, message.qp.startPsn);
    EXPECT_EQ(expected.qpn[0], message.qp.qpn[0]);
    EXPECT_EQ(expected.qpn[1], message.qp.qpn[1]);
}

TEST(RoceWireTest, ConnectFrameCarriesOneEntryPerQueuePair)
{
    QpInfo info = SampleQpInfo();
    info.totalQpNum = 3;
    info.qpn[2] = 303;
    std::vector<std::uint8_t> frame;
    ASSERT_TRUE(roce::EncodeConnectRequest(roce::kWireCmdConnectRequest, info, frame).ok());
    // Only the three queue pairs that were created are on the wire.
    EXPECT_EQ(roce::kWireHeaderBytes + 32U + 4U * 3U, frame.size());
    EXPECT_EQ(32U + 4U * 3U, ReadU32(frame, 12));

    Message message;
    std::string error;
    ASSERT_TRUE(roce::DecodeMessage(frame.data(), frame.size(), message, error).ok()) << error;
    EXPECT_EQ(3U, message.qp.totalQpNum);
    EXPECT_EQ(303U, message.qp.qpn[2]);
}

TEST(RoceWireTest, RejectsAConnectBodyThatDoesNotMatchTheQueuePairCount)
{
    QpInfo info = SampleQpInfo();
    std::vector<std::uint8_t> frame;
    ASSERT_TRUE(roce::EncodeConnectRequest(roce::kWireCmdConnectRequest, info, frame).ok());
    // Drop one queue pair entry but keep total_qp_num at two.
    frame.resize(frame.size() - 4U);
    WriteU32(frame, 12, static_cast<std::uint32_t>(frame.size() - roce::kWireHeaderBytes));
    WriteU32(frame, 0, Crc32(frame.data() + 4, frame.size() - 4));

    Message message;
    std::string error;
    EXPECT_FALSE(roce::DecodeMessage(frame.data(), frame.size(), message, error).ok());
    EXPECT_NE(std::string::npos, error.find("queue pair"));
}

TEST(RoceWireTest, AcceptsAnOlderCapabilityReplyWithoutDeleteKeyBatch)
{
    const Capabilities expected = SampleCapabilities();
    std::vector<std::uint8_t> frame = LegacyCapabilityReply(expected);

    Message message;
    std::string error;
    ASSERT_TRUE(roce::DecodeMessage(frame.data(), frame.size(), message, error).ok()) << error;
    EXPECT_TRUE(message.capabilityReply);
    EXPECT_EQ(expected.queueNum, message.capabilities.queueNum);
    EXPECT_EQ(expected.srMdts, message.capabilities.srMdts);
    EXPECT_EQ(expected.storeKeyBatch, message.capabilities.storeKeyBatch);
    EXPECT_EQ(expected.existKeyBatch, message.capabilities.existKeyBatch);
    EXPECT_EQ(expected.keyLength, message.capabilities.keyLength);
    EXPECT_EQ(expected.controllerCap, message.capabilities.controllerCap);
    // A field the peer did not send keeps its default instead of shifting the
    // ones behind it.
    EXPECT_EQ(Capabilities{}.deleteKeyBatch, message.capabilities.deleteKeyBatch);
}

TEST(RoceWireTest, ConnectRequestRejectsMoreQueuePairsThanTheTableHolds)
{
    QpInfo info = SampleQpInfo();
    info.totalQpNum = static_cast<std::uint8_t>(roce::kMaxQpPerLink + 1);
    std::vector<std::uint8_t> frame;
    const Status status = roce::EncodeConnectRequest(roce::kWireCmdConnectRequest, info, frame);
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, status.code);
}

TEST(RoceWireTest, ChecksumCoversTheHeaderAsWellAsTheBody)
{
    std::vector<std::uint8_t> frame;
    ASSERT_TRUE(roce::EncodeConnectDone(frame).ok());
    ASSERT_EQ(roce::kWireHeaderBytes, frame.size());

    // Flipping the command byte invalidates the CRC, which is exactly what the
    // specification asks for: everything except the crc field itself.
    std::vector<std::uint8_t> tampered = frame;
    tampered[5] = roce::kWireCmdDisconnect;
    Message message;
    std::string error;
    EXPECT_FALSE(roce::DecodeMessage(tampered.data(), tampered.size(), message, error).ok());
    EXPECT_NE(std::string::npos, error.find("checksum mismatch"));
}

TEST(RoceWireTest, ChecksumRejectsABodyThatWasEditedOnTheWire)
{
    std::vector<std::uint8_t> frame;
    ASSERT_TRUE(roce::EncodeCapabilityReply(SampleCapabilities(), frame).ok());
    frame[roce::kWireHeaderBytes + 40] ^= 0xFFU;

    Message message;
    std::string error;
    EXPECT_FALSE(roce::DecodeMessage(frame.data(), frame.size(), message, error).ok());
    EXPECT_NE(std::string::npos, error.find("checksum mismatch"));
}

TEST(RoceWireTest, RejectsShortFramesAndLengthMismatches)
{
    std::vector<std::uint8_t> frame;
    ASSERT_TRUE(roce::EncodeConnectDone(frame).ok());

    Message message;
    std::string error;
    EXPECT_FALSE(roce::DecodeMessage(frame.data(), 10, message, error).ok());
    EXPECT_NE(std::string::npos, error.find("shorter than the header"));

    // A frame that advertises 160 body bytes while carrying none must be refused
    // before the decoder starts reading fields.
    std::vector<std::uint8_t> lying = frame;
    lying[15] = 160;
    EXPECT_FALSE(roce::DecodeMessage(lying.data(), lying.size(), message, error).ok());
    EXPECT_NE(std::string::npos, error.find("does not match the frame"));
}

TEST(RoceWireTest, UnknownCommandsAndUnexpectedVersionsAreReported)
{
    std::vector<std::uint8_t> frame;
    ASSERT_TRUE(roce::EncodeDisconnect(frame).ok());
    ASSERT_EQ(roce::kWireHeaderBytes, frame.size());

    std::vector<std::uint8_t> unknown = frame;
    unknown[5] = 9;
    WriteU32(unknown, 0, Crc32(unknown.data() + 4, unknown.size() - 4));

    Message message;
    std::string error;
    EXPECT_FALSE(roce::DecodeMessage(unknown.data(), unknown.size(), message, error).ok());
    EXPECT_NE(std::string::npos, error.find("unknown command 9"));

    std::vector<std::uint8_t> newer = frame;
    newer[4] = static_cast<std::uint8_t>(roce::kWireVersion + 1);
    WriteU32(newer, 0, Crc32(newer.data() + 4, newer.size() - 4));
    EXPECT_FALSE(roce::DecodeMessage(newer.data(), newer.size(), message, error).ok());
    EXPECT_NE(std::string::npos, error.find("unsupported message version"));
}

TEST(RoceWireTest, MtuCodesFollowTheInfinibandScale)
{
    EXPECT_EQ(1U, roce::MtuCodeFromBytes(256));
    EXPECT_EQ(2U, roce::MtuCodeFromBytes(512));
    EXPECT_EQ(3U, roce::MtuCodeFromBytes(1024));
    EXPECT_EQ(4U, roce::MtuCodeFromBytes(2048));
    EXPECT_EQ(5U, roce::MtuCodeFromBytes(4096));
    EXPECT_EQ(4096U, roce::MtuBytesFromCode(5));
    EXPECT_EQ(0U, roce::MtuBytesFromCode(0));
}

}  // namespace
}  // namespace kv::test
