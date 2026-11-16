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
#include "cache/control_codec.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "status/status.h"
#include "type/block.h"

namespace UC::Store {
namespace {

BlockId MakeBlockId(std::uint32_t seed)
{
    BlockId id;
    for (std::size_t i = 0; i < id.bytes.size(); ++i) {
        id.bytes[i] = static_cast<std::byte>(seed * 31u + i);
    }
    return id;
}

ControlRequest MakeRequest(ControlOp op)
{
    ControlRequest request;
    request.op = op;
    request.seq = 42;
    request.blockIds = {MakeBlockId(1), MakeBlockId(2), MakeBlockId(3)};
    request.hasTtl = true;
    request.ttlMs = -9500;
    request.success = true;
    request.planId = std::uint64_t{0x0102030405060708};
    return request;
}

void ExpectRequestsEqual(const ControlRequest& actual, const ControlRequest& expected)
{
    EXPECT_EQ(actual.op, expected.op);
    EXPECT_EQ(actual.seq, expected.seq);
    ASSERT_EQ(actual.blockIds.size(), expected.blockIds.size());
    for (std::size_t i = 0; i < expected.blockIds.size(); ++i) {
        EXPECT_EQ(actual.blockIds[i], expected.blockIds[i]) << "block id " << i << " differs";
    }
    EXPECT_EQ(actual.hasTtl, expected.hasTtl);
    EXPECT_EQ(actual.ttlMs, expected.ttlMs);
    EXPECT_EQ(actual.success, expected.success);
    EXPECT_EQ(actual.planId, expected.planId);
}

std::vector<std::byte> Slice(const std::vector<std::byte>& data, std::size_t len)
{
    return {data.begin(), data.begin() + static_cast<std::ptrdiff_t>(len)};
}

TEST(ControlCodecTest, RequestRoundTripPerOp)
{
    const std::array<ControlOp, 8> ops = {
        ControlOp::Lookup,         ControlOp::Touch,          ControlOp::Pin,
        ControlOp::Unpin,          ControlOp::Alloc,          ControlOp::Commit,
        ControlOp::PrepareOffload, ControlOp::CompleteOffload};
    for (ControlOp op : ops) {
        const ControlRequest request = MakeRequest(op);
        auto encoded = EncodeRequest(request);
        ASSERT_TRUE(encoded.HasValue()) << encoded.Error().ToString();
        auto decoded = DecodeRequest(encoded.Value());
        ASSERT_TRUE(decoded.HasValue()) << decoded.Error().ToString();
        ExpectRequestsEqual(decoded.Value(), request);
    }
}

TEST(ControlCodecTest, RequestEmptyBlockIdsRoundTrip)
{
    ControlRequest request;
    request.op = ControlOp::Lookup;
    request.seq = 7;
    request.hasTtl = false;
    auto encoded = EncodeRequest(request);
    ASSERT_TRUE(encoded.HasValue());
    auto decoded = DecodeRequest(encoded.Value());
    ASSERT_TRUE(decoded.HasValue());
    ExpectRequestsEqual(decoded.Value(), request);
}

TEST(ControlCodecTest, RequestLargeBatchRoundTrip)
{
    ControlRequest request;
    request.op = ControlOp::Pin;
    request.seq = 9;
    request.blockIds.reserve(5000);
    for (std::uint32_t i = 0; i < 5000; ++i) { request.blockIds.push_back(MakeBlockId(i)); }
    auto encoded = EncodeRequest(request);
    ASSERT_TRUE(encoded.HasValue());
    EXPECT_EQ(encoded.Value().size(),
              27u + 16u * request.blockIds.size());  // header 27B + 16B per block id
    auto decoded = DecodeRequest(encoded.Value());
    ASSERT_TRUE(decoded.HasValue());
    ExpectRequestsEqual(decoded.Value(), request);
}

TEST(ControlCodecTest, RequestPlanIdTailIsLittleEndian)
{
    ControlRequest request;
    request.op = ControlOp::CompleteOffload;
    request.planId = std::uint64_t{0x0102030405060708};
    auto encoded = EncodeRequest(request);
    ASSERT_TRUE(encoded.HasValue());
    // Empty batch: op(1) + seq(4) + count(4) + hasTtl(1) + ttlMs(8) + success(1) = 19 bytes of
    // head, then the u64 planId tail, least significant byte first.
    ASSERT_EQ(encoded.Value().size(), 27u);
    const std::vector<std::byte> tail(encoded.Value().begin() + 19, encoded.Value().end());
    const std::vector<std::byte> expected{std::byte{0x08}, std::byte{0x07}, std::byte{0x06},
                                          std::byte{0x05}, std::byte{0x04}, std::byte{0x03},
                                          std::byte{0x02}, std::byte{0x01}};
    EXPECT_EQ(tail, expected);
}

TEST(ControlCodecTest, EncodeRequestRejectsHelloOp)
{
    ControlRequest request;
    request.op = ControlOp::Hello;
    auto encoded = EncodeRequest(request);
    ASSERT_FALSE(encoded.HasValue());
    EXPECT_EQ(encoded.Error(), Status::InvalidParam());
}

TEST(ControlCodecTest, DecodeRequestRejectsHelloBody)
{
    HelloRequest hello;
    hello.role = "worker";
    hello.uniqueId = "0123456789abcdef0123456789abcdef";
    hello.blockTotalNumber = 2048;
    hello.blockSizeBytes = 1048576;
    hello.replacementPolicy = "clock";
    hello.forceHugePage = true;
    auto encoded = EncodeHelloRequest(hello);
    ASSERT_TRUE(encoded.HasValue());
    auto decoded = DecodeRequest(encoded.Value());
    ASSERT_FALSE(decoded.HasValue());
    EXPECT_EQ(decoded.Error(), Status::InvalidParam());
}

TEST(ControlCodecTest, DecodeRequestRejectsUnknownOpcode)
{
    const std::vector<std::byte> bogus{std::byte{0xFF}, std::byte{0}, std::byte{0}, std::byte{0},
                                       std::byte{0},    std::byte{0}, std::byte{0}, std::byte{0},
                                       std::byte{0},    std::byte{0}, std::byte{0}, std::byte{0},
                                       std::byte{0},    std::byte{0}, std::byte{0}, std::byte{0},
                                       std::byte{0},    std::byte{0}, std::byte{0}};
    auto decoded = DecodeRequest(bogus);
    ASSERT_FALSE(decoded.HasValue());
    EXPECT_EQ(decoded.Error(), Status::InvalidParam());
}

TEST(ControlCodecTest, DecodeRequestRejectsEveryTruncation)
{
    auto encoded = EncodeRequest(MakeRequest(ControlOp::Alloc));
    ASSERT_TRUE(encoded.HasValue());
    for (std::size_t cut = 0; cut < encoded.Value().size(); ++cut) {
        auto decoded = DecodeRequest(Slice(encoded.Value(), cut));
        EXPECT_FALSE(decoded.HasValue()) << "truncation to " << cut << " was accepted";
    }
}

TEST(ControlCodecTest, DecodeRequestRejectsTrailingBytes)
{
    auto encoded = EncodeRequest(MakeRequest(ControlOp::Touch));
    ASSERT_TRUE(encoded.HasValue());
    std::vector<std::byte> padded = encoded.Value();
    padded.push_back(std::byte{0});
    auto decoded = DecodeRequest(padded);
    ASSERT_FALSE(decoded.HasValue());
    EXPECT_EQ(decoded.Error(), Status::InvalidParam());
}

TEST(ControlCodecTest, DecodeRequestRejectsCountMismatch)
{
    auto encoded = EncodeRequest(MakeRequest(ControlOp::Lookup));
    ASSERT_TRUE(encoded.HasValue());
    std::vector<std::byte> tampered = encoded.Value();
    tampered[5] = static_cast<std::byte>(static_cast<std::uint8_t>(tampered[5]) +
                                         1);  // count low byte: 3 blocks claimed, 2 delivered
    auto decoded = DecodeRequest(tampered);
    ASSERT_FALSE(decoded.HasValue());
    EXPECT_EQ(decoded.Error(), Status::InvalidParam());
}

TEST(ControlCodecTest, ResponseHitsRoundTrip)
{
    ControlResponse response;
    response.seq = 11;
    response.ok = true;
    response.statusCode = 0;
    response.message = "looked up";
    response.hits = {1, 0, 1, 1, 0};
    auto encoded = EncodeResponse(response);
    ASSERT_TRUE(encoded.HasValue());
    auto decoded = DecodeResponse(encoded.Value());
    ASSERT_TRUE(decoded.HasValue());
    const ControlResponse& out = decoded.Value();
    EXPECT_EQ(out.seq, response.seq);
    EXPECT_EQ(out.ok, response.ok);
    EXPECT_EQ(out.statusCode, response.statusCode);
    EXPECT_EQ(out.message, response.message);
    EXPECT_EQ(out.hits, response.hits);
    EXPECT_TRUE(out.slots.empty());
    EXPECT_TRUE(out.keys.empty());
    EXPECT_TRUE(out.tokenPayload.empty());
}

TEST(ControlCodecTest, ResponseSlotsWithTokenRoundTrip)
{
    ControlResponse response;
    response.seq = 12;
    response.ok = true;
    response.slots = {4, 5, 6};
    response.tokenPayload = {std::byte{1},  std::byte{0}, std::byte{0}, std::byte{0},
                             std::byte{64}, std::byte{0}, std::byte{0}, std::byte{0}};
    auto encoded = EncodeResponse(response);
    ASSERT_TRUE(encoded.HasValue());
    auto decoded = DecodeResponse(encoded.Value());
    ASSERT_TRUE(decoded.HasValue());
    const ControlResponse& out = decoded.Value();
    EXPECT_EQ(out.seq, response.seq);
    EXPECT_TRUE(out.ok);
    EXPECT_TRUE(out.hits.empty());
    EXPECT_EQ(out.slots, response.slots);
    EXPECT_TRUE(out.keys.empty());
    EXPECT_EQ(out.tokenPayload, response.tokenPayload);
}

TEST(ControlCodecTest, ResponseErrorRoundTrip)
{
    ControlResponse response;
    response.seq = 13;
    response.ok = false;
    response.statusCode = -50005;
    response.message = "block missing";
    auto encoded = EncodeResponse(response);
    ASSERT_TRUE(encoded.HasValue());
    auto decoded = DecodeResponse(encoded.Value());
    ASSERT_TRUE(decoded.HasValue());
    const ControlResponse& out = decoded.Value();
    EXPECT_EQ(out.seq, response.seq);
    EXPECT_FALSE(out.ok);
    EXPECT_EQ(out.statusCode, response.statusCode);
    EXPECT_EQ(out.message, response.message);
    EXPECT_TRUE(out.hits.empty());
    EXPECT_TRUE(out.slots.empty());
    EXPECT_TRUE(out.keys.empty());
    EXPECT_TRUE(out.tokenPayload.empty());
}

TEST(ControlCodecTest, ResponsePlanRoundTrip)
{
    ControlResponse response;
    response.seq = 14;
    response.ok = true;
    response.message = "planned";
    response.keys = {MakeBlockId(4), MakeBlockId(5)};
    // Token tail carries the plan payload (see PlanPayload): batchId 7, then
    // one source slot per key.
    response.tokenPayload = {std::byte{7}, std::byte{0}, std::byte{0}, std::byte{0},
                             std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
                             std::byte{2}, std::byte{0}, std::byte{0}, std::byte{0},
                             std::byte{9}, std::byte{0}, std::byte{0}, std::byte{0}};
    auto encoded = EncodeResponse(response);
    ASSERT_TRUE(encoded.HasValue());
    auto decoded = DecodeResponse(encoded.Value());
    ASSERT_TRUE(decoded.HasValue());
    const ControlResponse& out = decoded.Value();
    EXPECT_EQ(out.seq, response.seq);
    EXPECT_TRUE(out.ok);
    EXPECT_EQ(out.message, response.message);
    EXPECT_TRUE(out.hits.empty());
    EXPECT_TRUE(out.slots.empty());
    EXPECT_EQ(out.keys, response.keys);
    EXPECT_EQ(out.tokenPayload, response.tokenPayload);
}

TEST(ControlCodecTest, ResponseIdlePlanEncodesAsNone)
{
    // An idle PrepareOffload answers with no keys and no payload: the empty
    // plan falls back to the none kind instead of a bodyless plan kind, so
    // the frame is indistinguishable from any other bodyless success.
    ControlResponse response;
    response.seq = 15;
    response.ok = true;
    auto encoded = EncodeResponse(response);
    ASSERT_TRUE(encoded.HasValue());
    // seq(4) | ok(1) | statusCode(4) | msgLen(4) put the kind byte at index 13.
    EXPECT_EQ(encoded.Value()[13], std::byte{0});

    auto decoded = DecodeResponse(encoded.Value());
    ASSERT_TRUE(decoded.HasValue());
    EXPECT_TRUE(decoded.Value().keys.empty());
    EXPECT_TRUE(decoded.Value().tokenPayload.empty());
}

TEST(ControlCodecTest, EncodeResponseRejectsConflictingResultBodies)
{
    // The result body is single-choice: every pairing of hits, slots and
    // plan keys is rejected.
    ControlResponse response;
    response.hits = {1, 0};
    response.slots = {7};
    auto encoded = EncodeResponse(response);
    ASSERT_FALSE(encoded.HasValue());
    EXPECT_EQ(encoded.Error(), Status::InvalidParam());

    response.slots.clear();
    response.keys = {MakeBlockId(1)};
    encoded = EncodeResponse(response);
    ASSERT_FALSE(encoded.HasValue());
    EXPECT_EQ(encoded.Error(), Status::InvalidParam());

    response.hits.clear();
    response.slots = {7};
    encoded = EncodeResponse(response);
    ASSERT_FALSE(encoded.HasValue());
    EXPECT_EQ(encoded.Error(), Status::InvalidParam());
}

TEST(ControlCodecTest, DecodeResponseRejectsUnknownResultKind)
{
    // seq(4) | ok(1) | statusCode(4) | msgLen=0(4) | bogus kind
    const std::vector<std::byte> bogus{std::byte{0}, std::byte{0},   std::byte{0}, std::byte{13},
                                       std::byte{1}, std::byte{0},   std::byte{0}, std::byte{0},
                                       std::byte{0}, std::byte{0},   std::byte{0}, std::byte{0},
                                       std::byte{0}, std::byte{0x99}};
    auto decoded = DecodeResponse(bogus);
    ASSERT_FALSE(decoded.HasValue());
    EXPECT_EQ(decoded.Error(), Status::InvalidParam());
}

TEST(ControlCodecTest, DecodeResponseRejectsEveryTruncation)
{
    ControlResponse response;
    response.seq = 1;
    response.ok = true;
    response.message = "m";
    response.hits = {1, 1};
    auto encoded = EncodeResponse(response);
    ASSERT_TRUE(encoded.HasValue());
    for (std::size_t cut = 0; cut < encoded.Value().size(); ++cut) {
        auto decoded = DecodeResponse(Slice(encoded.Value(), cut));
        EXPECT_FALSE(decoded.HasValue()) << "hits truncation to " << cut << " was accepted";
    }

    // The plan kind's key body and plan payload must survive every cut too.
    ControlResponse plan;
    plan.seq = 2;
    plan.ok = true;
    plan.message = "m";
    plan.keys = {MakeBlockId(1), MakeBlockId(2)};
    plan.tokenPayload = std::vector<std::byte>(16, std::byte{0});
    auto encodedPlan = EncodeResponse(plan);
    ASSERT_TRUE(encodedPlan.HasValue());
    for (std::size_t cut = 0; cut < encodedPlan.Value().size(); ++cut) {
        auto decoded = DecodeResponse(Slice(encodedPlan.Value(), cut));
        EXPECT_FALSE(decoded.HasValue()) << "plan truncation to " << cut << " was accepted";
    }
}

TEST(ControlCodecTest, HelloRequestRoundTrip)
{
    HelloRequest hello;
    hello.role = "scheduler";
    hello.uniqueId = "0123456789abcdef0123456789abcdef";
    hello.blockTotalNumber = 131072;
    hello.blockSizeBytes = 3145728;
    hello.replacementPolicy = "fifo";
    hello.forceHugePage = true;
    auto encoded = EncodeHelloRequest(hello);
    ASSERT_TRUE(encoded.HasValue());
    auto decoded = DecodeHelloRequest(encoded.Value());
    ASSERT_TRUE(decoded.HasValue());
    const HelloRequest& out = decoded.Value();
    EXPECT_EQ(out.role, hello.role);
    EXPECT_EQ(out.uniqueId, hello.uniqueId);
    EXPECT_EQ(out.blockTotalNumber, hello.blockTotalNumber);
    EXPECT_EQ(out.blockSizeBytes, hello.blockSizeBytes);
    EXPECT_EQ(out.replacementPolicy, hello.replacementPolicy);
    EXPECT_EQ(out.forceHugePage, hello.forceHugePage);
}

TEST(ControlCodecTest, HelloRequestRejectsEveryTruncation)
{
    HelloRequest hello;
    hello.role = "worker";
    hello.uniqueId = "uid";
    hello.replacementPolicy = "lru";
    auto encoded = EncodeHelloRequest(hello);
    ASSERT_TRUE(encoded.HasValue());
    for (std::size_t cut = 0; cut < encoded.Value().size(); ++cut) {
        auto decoded = DecodeHelloRequest(Slice(encoded.Value(), cut));
        EXPECT_FALSE(decoded.HasValue()) << "truncation to " << cut << " was accepted";
    }
}

TEST(ControlCodecTest, DecodeHelloRequestRejectsWrongOp)
{
    auto encoded = EncodeRequest(MakeRequest(ControlOp::Lookup));
    ASSERT_TRUE(encoded.HasValue());
    auto decoded = DecodeHelloRequest(encoded.Value());
    ASSERT_FALSE(decoded.HasValue());
    EXPECT_EQ(decoded.Error(), Status::InvalidParam());
}

TEST(ControlCodecTest, HandshakeReplyIsPlainControlResponse)
{
    // A successful Hello answers ok=1 with no result body and no token; the
    // shared memfd rides out-of-band over SCM_RIGHTS and is invisible here.
    ControlResponse accepted;
    accepted.seq = 0;
    accepted.ok = true;
    auto encodedAck = EncodeResponse(accepted);
    ASSERT_TRUE(encodedAck.HasValue()) << encodedAck.Error().ToString();
    auto decodedAck = DecodeResponse(encodedAck.Value());
    ASSERT_TRUE(decodedAck.HasValue()) << decodedAck.Error().ToString();
    const ControlResponse& ack = decodedAck.Value();
    EXPECT_EQ(ack.seq, 0u);
    EXPECT_TRUE(ack.ok);
    EXPECT_TRUE(ack.message.empty());
    EXPECT_TRUE(ack.hits.empty());
    EXPECT_TRUE(ack.slots.empty());
    EXPECT_TRUE(ack.keys.empty());
    EXPECT_TRUE(ack.tokenPayload.empty());

    // A rejected Hello answers ok=0 with the leader's reason.
    ControlResponse rejected;
    rejected.seq = 0;
    rejected.ok = false;
    rejected.statusCode = Status::InvalidParam().Underlying();
    rejected.message = "handshake digest mismatch";
    auto encodedErr = EncodeResponse(rejected);
    ASSERT_TRUE(encodedErr.HasValue()) << encodedErr.Error().ToString();
    auto decodedErr = DecodeResponse(encodedErr.Value());
    ASSERT_TRUE(decodedErr.HasValue()) << decodedErr.Error().ToString();
    const ControlResponse& err = decodedErr.Value();
    EXPECT_EQ(err.seq, 0u);
    EXPECT_FALSE(err.ok);
    EXPECT_EQ(err.statusCode, rejected.statusCode);
    EXPECT_EQ(err.message, rejected.message);
}

}  // namespace
}  // namespace UC::Store
