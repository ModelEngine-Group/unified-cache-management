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
#include "kv_protocol.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include "dram/dram_test_common.h"

namespace UC::DramPool {
namespace {

using UC::Test::Dram::KeyFromHex;
constexpr std::uint64_t kRequestId = 0x123456789ABCDEF0ULL;

void ExpectLe16(const std::vector<std::uint8_t>& buf, std::size_t offset, std::uint16_t value)
{
    EXPECT_EQ(buf[offset], static_cast<std::uint8_t>(value & 0xFF));
    EXPECT_EQ(buf[offset + 1], static_cast<std::uint8_t>((value >> 8) & 0xFF));
}

void ExpectLe32(const std::vector<std::uint8_t>& buf, std::size_t offset, std::uint32_t value)
{
    for (std::size_t i = 0; i < sizeof(value); ++i) {
        EXPECT_EQ(buf[offset + i], static_cast<std::uint8_t>((value >> (i * 8)) & 0xFF));
    }
}

void ExpectLe64(const std::vector<std::uint8_t>& buf, std::size_t offset, std::uint64_t value)
{
    for (std::size_t i = 0; i < sizeof(value); ++i) {
        EXPECT_EQ(buf[offset + i], static_cast<std::uint8_t>((value >> (i * 8)) & 0xFF));
    }
}

class UCDramKvProtocolTest : public ::testing::Test {
protected:
    ProtocolManager mgr_;
};

TEST(UCDramKvProtocolOpcodeTest, UsesDramStoreOpType)
{
    EXPECT_EQ(static_cast<std::uint8_t>(OpType::LOOKUP), 0x0);
    EXPECT_EQ(static_cast<std::uint8_t>(OpType::DUMP), 0x1);
    EXPECT_EQ(static_cast<std::uint8_t>(OpType::LOAD), 0x2);
}

TEST_F(UCDramKvProtocolTest, PackDumpRequestMatchesLayout)
{
    KvDumpEntry entry;
    entry.key = KeyFromHex("10");
    entry.addr = 0x1122334455667788ULL;
    entry.len = 0xAABBCCDD;
    entry.idx = 0x12345678;

    KvDumpRequest req;
    req.opcode = OpType::DUMP;
    req.request_id = kRequestId;
    req.resp_addr = 0x0102030405060708ULL;
    req.batch_size = 1;
    req.ttl = 0x99AABBCCU;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    ASSERT_TRUE(status.Success()) << status.ToString();

    EXPECT_EQ(packed[0], static_cast<std::uint8_t>(OpType::DUMP));
    ExpectLe64(packed, kRequestIdOffset, req.request_id);
    ExpectLe64(packed, kRespAddrOffset, req.resp_addr);
    ExpectLe32(packed, kDumpTtlOffset, req.ttl);
    ExpectLe16(packed, kDumpBatchSizeOffset, req.batch_size);
    EXPECT_EQ(packed.size(), kKvDumpRequestHeaderSize + kKvDumpEntrySize);
    EXPECT_EQ(std::memcmp(packed.data() + kKvDumpRequestHeaderSize, entry.key.data(), kKvKeySize),
              0);
    ExpectLe64(packed, kKvDumpRequestHeaderSize + 16, entry.addr);
    ExpectLe32(packed, kKvDumpRequestHeaderSize + 24, entry.len);
    ExpectLe32(packed, kKvDumpRequestHeaderSize + 28, entry.idx);
}

TEST_F(UCDramKvProtocolTest, PackLoadRequestMatchesLayout)
{
    KvLoadEntry entry;
    entry.key = KeyFromHex("20");
    entry.addr = 0x8877665544332211ULL;
    entry.len = 0x1000;
    entry.idx = 7;

    KvLoadRequest req;
    req.opcode = OpType::LOAD;
    req.request_id = kRequestId;
    req.resp_addr = 0x1111222233334444ULL;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    ASSERT_TRUE(status.Success()) << status.ToString();

    EXPECT_EQ(packed[0], static_cast<std::uint8_t>(OpType::LOAD));
    ExpectLe64(packed, kRequestIdOffset, req.request_id);
    ExpectLe64(packed, kRespAddrOffset, req.resp_addr);
    ExpectLe16(packed, kLoadLookupBatchSizeOffset, req.batch_size);
    EXPECT_EQ(std::memcmp(packed.data() + kKvLoadRequestHeaderSize, entry.key.data(), kKvKeySize),
              0);
}

TEST_F(UCDramKvProtocolTest, PackLookupRequestMatchesLayout)
{
    KvLookupEntry entry0;
    entry0.key = KeyFromHex("30");
    KvLookupEntry entry1;
    entry1.key = KeyFromHex("40");

    KvLookupRequest req;
    req.opcode = OpType::LOOKUP;
    req.request_id = kRequestId;
    req.resp_addr = 0x1234000056780000ULL;
    req.batch_size = 2;
    req.entries = {entry0, entry1};

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    ASSERT_TRUE(status.Success()) << status.ToString();

    EXPECT_EQ(packed[0], static_cast<std::uint8_t>(OpType::LOOKUP));
    ExpectLe64(packed, kRequestIdOffset, req.request_id);
    ExpectLe64(packed, kRespAddrOffset, req.resp_addr);
    ExpectLe16(packed, kLoadLookupBatchSizeOffset, req.batch_size);
    EXPECT_EQ(
        std::memcmp(packed.data() + kKvLookupRequestHeaderSize, entry0.key.data(), kKvKeySize), 0);
    EXPECT_EQ(std::memcmp(packed.data() + kKvLookupRequestHeaderSize + kKvLookupEntrySize,
                          entry1.key.data(), kKvKeySize),
              0);
}

TEST_F(UCDramKvProtocolTest, RejectsBatchSizeMismatch)
{
    KvLookupEntry entry;
    entry.key = KeyFromHex("50");

    KvLookupRequest req;
    req.opcode = OpType::LOOKUP;
    req.request_id = kRequestId;
    req.resp_addr = 0x1000;
    req.batch_size = 2;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(kKvLookupRequestHeaderSize + 2 * kKvLookupEntrySize, 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("batch_size"), std::string::npos);
}

TEST_F(UCDramKvProtocolTest, RejectsAllZeroKey)
{
    KvLookupEntry entry;

    KvLookupRequest req;
    req.opcode = OpType::LOOKUP;
    req.request_id = kRequestId;
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("key"), std::string::npos);
}

TEST_F(UCDramKvProtocolTest, RejectsZeroDumpLoadAddrAndLen)
{
    KvDumpEntry entry;
    entry.key = KeyFromHex("60");
    entry.addr = 0;
    entry.len = 1;

    KvDumpRequest req;
    req.opcode = OpType::DUMP;
    req.request_id = kRequestId;
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("addr"), std::string::npos);

    req.entries[0].addr = 0x2000;
    req.entries[0].len = 0;
    status = mgr_.PackRequest(packed.data(), req.opcode, req);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("len"), std::string::npos);
}

TEST_F(UCDramKvProtocolTest, UnpackRequestRejectsWrongSize)
{
    KvLookupEntry entry;
    entry.key = KeyFromHex("70");

    KvLookupRequest req;
    req.opcode = OpType::LOOKUP;
    req.request_id = kRequestId;
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    ASSERT_TRUE(status.Success()) << status.ToString();

    std::unique_ptr<KvRequest> parsed;
    status = mgr_.UnpackRequest(packed.data(), packed.size() - 1, parsed);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("size"), std::string::npos);
}

TEST_F(UCDramKvProtocolTest, UnpackResponseReadsPackedResults)
{
    std::uint8_t lookupFlag[] = {0xF0, 0xDE, 0xBC,
                                 0x9A, 0x78, 0x56,
                                 0x34, 0x12, static_cast<std::uint8_t>(ResponseStatus::Ready),
                                 0x8D, 0x01};
    KvResponse resp;
    auto status = mgr_.UnpackResponse(lookupFlag, OpType::LOOKUP, kRequestId, 9, resp);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(resp.results, (std::vector<std::uint8_t>{1, 0, 1, 1, 0, 0, 0, 1, 1}));

    std::uint8_t dumpFlag[] = {0xF0, 0xDE, 0xBC,
                               0x9A, 0x78, 0x56,
                               0x34, 0x12, static_cast<std::uint8_t>(ResponseStatus::Ready),
                               0x10, 0xF2, 0x03};
    resp.results.clear();
    status = mgr_.UnpackResponse(dumpFlag, OpType::DUMP, kRequestId, 5, resp);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(resp.results, (std::vector<std::uint8_t>{0, 1, 2, 15, 3}));
}

TEST_F(UCDramKvProtocolTest, ServerRoundTripDumpLoad)
{
    KvLoadEntry entry;
    entry.key = KeyFromHex("80");
    entry.addr = 0xAABBCCDDEEFF0011ULL;
    entry.len = 0x2000;
    entry.idx = 0x55;

    KvLoadRequest req;
    req.opcode = OpType::LOAD;
    req.request_id = kRequestId;
    req.resp_addr = 0x9988776655443322ULL;
    req.batch_size = 1;
    req.entries = {entry};

    // client packs
    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).Success());

    // server unpacks (validation merged into UnpackRequest)
    std::unique_ptr<KvRequest> parsed;
    ASSERT_TRUE(mgr_.UnpackRequest(packed.data(), packed.size(), parsed).Success());
    auto& dl = static_cast<KvLoadRequest&>(*parsed);
    EXPECT_EQ(dl.opcode, req.opcode);
    EXPECT_EQ(dl.request_id, req.request_id);
    EXPECT_EQ(dl.resp_addr, req.resp_addr);
    EXPECT_EQ(dl.batch_size, req.batch_size);
    EXPECT_EQ(std::memcmp(dl.entries[0].key.data(), entry.key.data(), kKvKeySize), 0);
    EXPECT_EQ(dl.entries[0].addr, entry.addr);
    EXPECT_EQ(dl.entries[0].len, entry.len);
    EXPECT_EQ(dl.entries[0].idx, entry.idx);

    // server packs response, client unpacks it
    KvResponse resp;
    resp.request_id = req.request_id;
    resp.results = {0x0};  // batch_size == 1 errcode
    std::vector<std::uint8_t> flag(mgr_.GetPackedResponseSize(req.opcode, resp.results.size()),
                                   0xFF);
    ASSERT_TRUE(mgr_.PackResponse(flag.data(), req.opcode, resp).Success());
    KvResponse resp2;
    ASSERT_TRUE(mgr_.UnpackResponse(flag.data(), req.opcode, req.request_id, req.batch_size, resp2)
                    .Success());
    EXPECT_EQ(resp2.request_id, req.request_id);
    ASSERT_EQ(resp2.results.size(), 1u);
    EXPECT_EQ(resp2.results[0], 0x0U);
}

TEST_F(UCDramKvProtocolTest, ServerRoundTripLookup)
{
    KvLookupEntry e0;
    e0.key = KeyFromHex("90");
    KvLookupEntry e1;
    e1.key = KeyFromHex("a0");
    KvLookupRequest req;
    req.opcode = OpType::LOOKUP;
    req.request_id = kRequestId;
    req.resp_addr = 0x0E0E0E0E0E0E0E0EULL;
    req.batch_size = 2;
    req.entries = {e0, e1};

    // client packs
    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).Success());

    // server verifies + unpacks
    std::unique_ptr<KvRequest> parsed;
    ASSERT_TRUE(mgr_.UnpackRequest(packed.data(), packed.size(), parsed).Success());
    auto& lk = static_cast<KvLookupRequest&>(*parsed);
    EXPECT_EQ(lk.opcode, req.opcode);
    EXPECT_EQ(lk.request_id, req.request_id);
    EXPECT_EQ(lk.resp_addr, req.resp_addr);
    EXPECT_EQ(lk.batch_size, req.batch_size);
    EXPECT_EQ(std::memcmp(lk.entries[0].key.data(), e0.key.data(), kKvKeySize), 0);
    EXPECT_EQ(std::memcmp(lk.entries[1].key.data(), e1.key.data(), kKvKeySize), 0);

    // server packs one existence bit per key, client unpacks it
    KvResponse resp;
    resp.request_id = req.request_id;
    resp.results = {1, 0};
    std::vector<std::uint8_t> flag(mgr_.GetPackedResponseSize(req.opcode, resp.results.size()),
                                   0xFF);
    ASSERT_TRUE(mgr_.PackResponse(flag.data(), req.opcode, resp).Success());
    KvResponse resp2;
    ASSERT_TRUE(mgr_.UnpackResponse(flag.data(), req.opcode, req.request_id, req.batch_size, resp2)
                    .Success());
    EXPECT_EQ(resp2.results, resp.results);
}

// ---------------------------------------------------------------------------
// Boundary values: every field set to max
// ---------------------------------------------------------------------------

TEST_F(UCDramKvProtocolTest, DumpLoadMaxFieldValuesRoundTrip)
{
    KvLoadEntry entry;
    entry.key.fill(std::byte{0xFF});
    entry.addr = 0xFFFFFFFFFFFFFFFFULL;
    entry.len = 0xFFFFFFFFU;
    entry.idx = 0xFFFFFFFFU;

    KvLoadRequest req;
    req.opcode = OpType::LOAD;
    req.request_id = std::numeric_limits<std::uint64_t>::max();
    req.resp_addr = 0xFFFFFFFFFFFFFFFFULL;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).Success());

    std::unique_ptr<KvRequest> parsed;
    ASSERT_TRUE(mgr_.UnpackRequest(packed.data(), packed.size(), parsed).Success());
    auto& dl = static_cast<KvLoadRequest&>(*parsed);
    EXPECT_EQ(dl.request_id, std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(dl.resp_addr, 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_EQ(dl.entries[0].addr, 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_EQ(dl.entries[0].len, 0xFFFFFFFFU);
    EXPECT_EQ(dl.entries[0].idx, 0xFFFFFFFFU);
    EXPECT_EQ(std::memcmp(dl.entries[0].key.data(), entry.key.data(), kKvKeySize), 0);
}

// ---------------------------------------------------------------------------
// Multi-entry: batch_size > 1, verify every entry survives the round-trip
// ---------------------------------------------------------------------------

TEST_F(UCDramKvProtocolTest, DumpLoadMultiEntryRoundTrip)
{
    constexpr std::uint16_t kBatch = 5;
    KvDumpRequest req;
    req.opcode = OpType::DUMP;
    req.request_id = kRequestId;
    req.resp_addr = 0xA5A5A5A5A5A5A5A5ULL;
    req.batch_size = kBatch;
    req.ttl = 0x10203040U;
    for (std::uint16_t i = 0; i < kBatch; ++i) {
        KvDumpEntry e;
        e.key = KeyFromHex(std::to_string(i + 1U).c_str());
        e.addr = 0x1000ULL * (i + 1);
        e.len = 0x200U * (i + 1);
        e.idx = i;
        req.entries.push_back(e);
    }

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).Success());

    std::unique_ptr<KvRequest> parsed;
    ASSERT_TRUE(mgr_.UnpackRequest(packed.data(), packed.size(), parsed).Success());
    auto& dl = static_cast<KvDumpRequest&>(*parsed);
    EXPECT_EQ(dl.ttl, req.ttl);
    ASSERT_EQ(dl.entries.size(), kBatch);
    for (std::uint16_t i = 0; i < kBatch; ++i) {
        EXPECT_EQ(std::memcmp(dl.entries[i].key.data(), req.entries[i].key.data(), kKvKeySize), 0)
            << "entry " << i;
        EXPECT_EQ(dl.entries[i].addr, req.entries[i].addr) << "entry " << i;
        EXPECT_EQ(dl.entries[i].len, req.entries[i].len) << "entry " << i;
        EXPECT_EQ(dl.entries[i].idx, req.entries[i].idx) << "entry " << i;
    }
}

TEST_F(UCDramKvProtocolTest, LookupMultiEntryRoundTrip)
{
    constexpr std::uint16_t kBatch = 4;
    KvLookupRequest req;
    req.opcode = OpType::LOOKUP;
    req.request_id = kRequestId;
    req.resp_addr = 0xB0B0B0B0B0B0B0B0ULL;
    req.batch_size = kBatch;
    for (std::uint16_t i = 0; i < kBatch; ++i) {
        KvLookupEntry e;
        e.key = KeyFromHex(std::to_string(i + 1U).c_str());
        req.entries.push_back(e);
    }

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).Success());

    std::unique_ptr<KvRequest> parsed;
    ASSERT_TRUE(mgr_.UnpackRequest(packed.data(), packed.size(), parsed).Success());
    auto& lk = static_cast<KvLookupRequest&>(*parsed);
    ASSERT_EQ(lk.entries.size(), kBatch);
    for (std::uint16_t i = 0; i < kBatch; ++i) {
        EXPECT_EQ(std::memcmp(lk.entries[i].key.data(), req.entries[i].key.data(), kKvKeySize), 0)
            << "entry " << i;
    }
}

// ---------------------------------------------------------------------------
// opcode validation
// ---------------------------------------------------------------------------

TEST_F(UCDramKvProtocolTest, RejectsUnknownOpcodeOnDumpLoad)
{
    KvDumpRequest req;
    const auto unknownOpcode = static_cast<OpType>(0xFF);
    req.opcode = unknownOpcode;
    req.request_id = kRequestId;
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {
        KvDumpEntry{KeyFromHex("10"), 0x2000, 0x100, 0}
    };

    std::vector<std::uint8_t> buf(mgr_.GetPackedRequestSize(OpType::DUMP, req), 0);
    auto status = mgr_.PackRequest(buf.data(), unknownOpcode, req);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("opcode"), std::string::npos);
}

TEST_F(UCDramKvProtocolTest, RejectsOpcodeMismatch)
{
    KvDumpRequest req;
    req.opcode = OpType::LOOKUP;  // wrong opcode for a Dump request
    req.request_id = kRequestId;
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {
        KvDumpEntry{KeyFromHex("10"), 0x2000, 0x100, 0}
    };

    std::vector<std::uint8_t> buf(mgr_.GetPackedRequestSize(OpType::DUMP, req), 0);
    auto status = mgr_.PackRequest(buf.data(), OpType::DUMP, req);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("opcode"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Server UnpackRequest edge cases (validation is now merged into UnpackRequest)
// ---------------------------------------------------------------------------

TEST_F(UCDramKvProtocolTest, UnpackRequestRejectsUnknownOpcode)
{
    std::vector<std::uint8_t> buf(kKvLookupRequestHeaderSize + kKvLookupEntrySize, 0);
    buf[0] = 0xEE;  // unknown opcode
    std::unique_ptr<KvRequest> parsed;
    auto status = mgr_.UnpackRequest(buf.data(), buf.size(), parsed);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("unknown opcode"), std::string::npos);
}

TEST_F(UCDramKvProtocolTest, UnpackRequestRejectsExtraBytes)
{
    KvLookupRequest req;
    req.opcode = OpType::LOOKUP;
    req.request_id = kRequestId;
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {KvLookupEntry{KeyFromHex("70")}};

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).Success());
    packed.push_back(0x00);  // one extra byte
    std::unique_ptr<KvRequest> parsed;
    auto status = mgr_.UnpackRequest(packed.data(), packed.size(), parsed);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("size"), std::string::npos);
}

TEST_F(UCDramKvProtocolTest, UnpackRequestRejectsNull)
{
    std::unique_ptr<KvRequest> out;
    auto status = mgr_.UnpackRequest(nullptr, kKvLoadRequestHeaderSize, out);
    EXPECT_FALSE(status.Success());
}

TEST_F(UCDramKvProtocolTest, UnpackRequestRejectsTruncatedHeaderForEveryOpcode)
{
    struct TestCase {
        OpType opcode;
        std::size_t header_size;
    };
    const std::array<TestCase, 3> test_cases = {
        {{OpType::DUMP, kKvDumpRequestHeaderSize},
         {OpType::LOAD, kKvLoadRequestHeaderSize},
         {OpType::LOOKUP, kKvLookupRequestHeaderSize}}
    };

    for (const auto& test_case : test_cases) {
        SCOPED_TRACE(static_cast<std::uint8_t>(test_case.opcode));
        std::vector<std::uint8_t> buf(test_case.header_size - 1, 0);
        buf[0] = static_cast<std::uint8_t>(test_case.opcode);
        std::unique_ptr<KvRequest> out;
        auto status = mgr_.UnpackRequest(buf.data(), buf.size(), out);
        EXPECT_FALSE(status.Success());
        EXPECT_NE(status.ToString().find("header"), std::string::npos);
    }
}

TEST_F(UCDramKvProtocolTest, UnpackRequestRejectsSizeMismatch)
{
    KvDumpRequest req;
    req.opcode = OpType::DUMP;
    req.request_id = kRequestId;
    req.resp_addr = 0x1000;
    req.batch_size = 2;
    req.entries = {
        KvDumpEntry{KeyFromHex("10"), 0x2000, 0x100, 0},
        KvDumpEntry{KeyFromHex("20"), 0x3000, 0x200, 1}
    };

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).Success());
    std::unique_ptr<KvRequest> out;
    // truncated by 1 byte
    auto status = mgr_.UnpackRequest(packed.data(), packed.size() - 1, out);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("size"), std::string::npos);
}

// ---------------------------------------------------------------------------
// PackResponse edge cases
// ---------------------------------------------------------------------------

TEST_F(UCDramKvProtocolTest, PackResponseRejectsNullData)
{
    KvResponse resp;
    resp.request_id = kRequestId;
    resp.results = {0x0};
    auto status = mgr_.PackResponse(nullptr, OpType::DUMP, resp);
    EXPECT_FALSE(status.Success());
}

TEST_F(UCDramKvProtocolTest, PackResponseRejectsValuesThatDoNotFitWireWidth)
{
    std::uint8_t flag[kResponseResultsOffset + 1] = {0};
    KvResponse lookup;
    lookup.request_id = kRequestId;
    lookup.results = {2};
    auto status = mgr_.PackResponse(flag, OpType::LOOKUP, lookup);
    EXPECT_FALSE(status.Success()) << status.ToString();
    EXPECT_EQ(flag[kResponseStatusOffset], static_cast<std::uint8_t>(ResponseStatus::Pending));

    KvResponse dump;
    dump.request_id = kRequestId;
    dump.results = {16};
    status = mgr_.PackResponse(flag, OpType::DUMP, dump);
    EXPECT_FALSE(status.Success()) << status.ToString();
    EXPECT_EQ(flag[kResponseStatusOffset], static_cast<std::uint8_t>(ResponseStatus::Pending));
}

TEST_F(UCDramKvProtocolTest, PackResponseLookupRejectsZeroCount)
{
    KvResponse resp;  // empty results
    resp.request_id = kRequestId;
    std::uint8_t flag[kResponseResultsOffset] = {0};
    auto status = mgr_.PackResponse(flag, OpType::LOOKUP, resp);
    EXPECT_FALSE(status.Success());
}

TEST_F(UCDramKvProtocolTest, PackResponseDumpLoadZeroErrcodes)
{
    KvResponse resp;  // empty, result_count=0
    resp.request_id = kRequestId;
    std::uint8_t flag[kResponseResultsOffset] = {0xFF};
    auto status = mgr_.PackResponse(flag, OpType::DUMP, resp);
    EXPECT_TRUE(status.Success());
    EXPECT_EQ(flag[kResponseStatusOffset], static_cast<std::uint8_t>(ResponseStatus::Ready));
}

// ---------------------------------------------------------------------------
// UnpackResponse edge cases
// ---------------------------------------------------------------------------

TEST_F(UCDramKvProtocolTest, UnpackResponseRejectsNullData)
{
    KvResponse resp;
    auto status = mgr_.UnpackResponse(nullptr, OpType::DUMP, kRequestId, 1, resp);
    EXPECT_FALSE(status.Success());
}

TEST_F(UCDramKvProtocolTest, ReportsPendingAndReadyResponseStatus)
{
    bool ready = true;
    const std::uint8_t pending[kResponseResultsOffset] = {};
    auto status = mgr_.IsResponseReady(pending, kRequestId, ready);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_FALSE(ready);

    std::uint8_t completed[kResponseResultsOffset] = {};
    std::memcpy(completed + kResponseRequestIdOffset, &kRequestId, sizeof(kRequestId));
    completed[kResponseStatusOffset] = static_cast<std::uint8_t>(ResponseStatus::Ready);
    status = mgr_.IsResponseReady(completed, kRequestId, ready);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_TRUE(ready);
}

TEST_F(UCDramKvProtocolTest, ReadyResponseWithDifferentRequestIdReturnsError)
{
    std::uint8_t response[kResponseResultsOffset] = {};
    const std::uint64_t staleRequestId = kRequestId - 1U;
    std::memcpy(response + kResponseRequestIdOffset, &staleRequestId, sizeof(staleRequestId));
    response[kResponseStatusOffset] = static_cast<std::uint8_t>(ResponseStatus::Ready);

    bool ready = true;
    const auto status = mgr_.IsResponseReady(response, kRequestId, ready);

    EXPECT_TRUE(status.Failure());
    EXPECT_FALSE(ready);
}

TEST_F(UCDramKvProtocolTest, ResponseStatusRejectsNullAndUnknownValues)
{
    bool ready = true;
    auto status = mgr_.IsResponseReady(nullptr, kRequestId, ready);
    EXPECT_TRUE(status.Failure());
    EXPECT_FALSE(ready);

    std::uint8_t invalid[kResponseResultsOffset] = {};
    invalid[kResponseStatusOffset] = 2;
    ready = true;
    status = mgr_.IsResponseReady(invalid, kRequestId, ready);
    EXPECT_TRUE(status.Failure());
    EXPECT_FALSE(ready);
}

TEST_F(UCDramKvProtocolTest, UnpackResponseReturnsRetryWithoutChangingResultsWhilePending)
{
    std::uint8_t pending[kResponseResultsOffset + 1] = {};
    pending[kResponseResultsOffset] = 0xFF;
    KvResponse response;
    response.results = {7, 8};

    const auto status = mgr_.UnpackResponse(pending, OpType::DUMP, kRequestId, 2, response);

    EXPECT_EQ(status, Status::Retry());
    EXPECT_EQ(response.results, (std::vector<std::uint8_t>{7, 8}));
}

TEST_F(UCDramKvProtocolTest, PackedResponseSizesRoundUpAtBitBoundaries)
{
    EXPECT_EQ(mgr_.GetPackedResponseSize(OpType::LOOKUP, 1), 10U);
    EXPECT_EQ(mgr_.GetPackedResponseSize(OpType::LOOKUP, 8), 10U);
    EXPECT_EQ(mgr_.GetPackedResponseSize(OpType::LOOKUP, 9), 11U);
    EXPECT_EQ(mgr_.GetPackedResponseSize(OpType::DUMP, 1), 10U);
    EXPECT_EQ(mgr_.GetPackedResponseSize(OpType::DUMP, 2), 10U);
    EXPECT_EQ(mgr_.GetPackedResponseSize(OpType::LOAD, 3), 11U);
    EXPECT_EQ(mgr_.GetPackedResponseSize(static_cast<OpType>(0xFF), 3), 0U);
}

// ---------------------------------------------------------------------------
// Full response symmetry: PackResponse -> UnpackResponse exact match
// ---------------------------------------------------------------------------

TEST_F(UCDramKvProtocolTest, ResponseSymmetryMultipleErrcodes)
{
    KvResponse resp;
    resp.request_id = kRequestId;
    resp.results = {0x0, 0x1, 0x2, 0xE, 0xF};
    constexpr std::uint16_t kCount = 5;
    std::vector<std::uint8_t> flag(mgr_.GetPackedResponseSize(OpType::DUMP, kCount), 0xFF);

    ASSERT_TRUE(mgr_.PackResponse(flag.data(), OpType::DUMP, resp).Success());
    EXPECT_EQ(flag, (std::vector<std::uint8_t>{0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12,
                                               static_cast<std::uint8_t>(ResponseStatus::Ready),
                                               0x10, 0xE2, 0x0F}));
    KvResponse resp2;
    ASSERT_TRUE(
        mgr_.UnpackResponse(flag.data(), OpType::DUMP, kRequestId, kCount, resp2).Success());
    ASSERT_EQ(resp2.results.size(), kCount);
    for (std::uint16_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(resp2.results[i], resp.results[i]) << "result " << i;
    }
}

// ---------------------------------------------------------------------------
// Multi-round: same manager handles many sequential operations
// ---------------------------------------------------------------------------

TEST_F(UCDramKvProtocolTest, MultiRoundSequentialPacks)
{
    for (std::uint8_t round = 0; round < 10; ++round) {
        const auto opcode = (round % 2 == 0) ? OpType::DUMP : OpType::LOAD;
        const auto resp_addr = 0x1000ULL * (round + 1);
        const auto addr = 0x2000ULL * (round + 1);
        const auto len = 0x100U * (round + 1);

        std::vector<std::uint8_t> packed;
        if (opcode == OpType::DUMP) {
            KvDumpRequest req;
            req.opcode = opcode;
            req.request_id = static_cast<std::uint64_t>(round) + 1U;
            req.resp_addr = resp_addr;
            req.batch_size = 1;
            req.ttl = 0x500U * (round + 1);
            req.entries = {
                KvDumpEntry{KeyFromHex(std::to_string(round + 1U).c_str()), addr, len, round}
            };
            packed.resize(mgr_.GetPackedRequestSize(opcode, req), 0);
            ASSERT_TRUE(mgr_.PackRequest(packed.data(), opcode, req).Success())
                << "round " << round;
        } else {
            KvLoadRequest req;
            req.opcode = opcode;
            req.request_id = static_cast<std::uint64_t>(round) + 1U;
            req.resp_addr = resp_addr;
            req.batch_size = 1;
            req.entries = {
                KvLoadEntry{KeyFromHex(std::to_string(round + 1U).c_str()), addr, len, round}
            };
            packed.resize(mgr_.GetPackedRequestSize(opcode, req), 0);
            ASSERT_TRUE(mgr_.PackRequest(packed.data(), opcode, req).Success())
                << "round " << round;
        }

        std::unique_ptr<KvRequest> parsed;
        ASSERT_TRUE(mgr_.UnpackRequest(packed.data(), packed.size(), parsed).Success())
            << "round " << round;
        if (opcode == OpType::DUMP) {
            auto& dl = static_cast<KvDumpRequest&>(*parsed);
            EXPECT_EQ(dl.opcode, opcode) << "round " << round;
            EXPECT_EQ(dl.resp_addr, resp_addr) << "round " << round;
            EXPECT_EQ(dl.ttl, 0x500U * (round + 1)) << "round " << round;
            EXPECT_EQ(dl.entries[0].idx, round) << "round " << round;
        } else {
            auto& dl = static_cast<KvLoadRequest&>(*parsed);
            EXPECT_EQ(dl.opcode, opcode) << "round " << round;
            EXPECT_EQ(dl.resp_addr, resp_addr) << "round " << round;
            EXPECT_EQ(dl.entries[0].idx, round) << "round " << round;
        }

        // response round-trip each iteration
        KvResponse resp;
        resp.request_id = static_cast<std::uint64_t>(round) + 1U;
        resp.results = {round};
        std::uint8_t flag[kResponseResultsOffset + 1] = {};
        ASSERT_TRUE(mgr_.PackResponse(flag, opcode, resp).Success()) << "round " << round;
        KvResponse resp2;
        ASSERT_TRUE(mgr_.UnpackResponse(flag, opcode, resp.request_id, 1, resp2).Success())
            << "round " << round;
        EXPECT_EQ(resp2.results[0], round) << "round " << round;
    }
}

// ---------------------------------------------------------------------------
// Full client-server round-trip with response values spanning the range
// ---------------------------------------------------------------------------

TEST_F(UCDramKvProtocolTest, LookupPackedResponseOverwritesNonZeroBufferAndRoundTrips)
{
    KvResponse resp;
    resp.request_id = kRequestId;
    resp.results = {1, 0, 1, 1, 0, 0, 0, 1, 1};
    std::vector<std::uint8_t> flag(mgr_.GetPackedResponseSize(OpType::LOOKUP, resp.results.size()),
                                   0xFF);

    ASSERT_TRUE(mgr_.PackResponse(flag.data(), OpType::LOOKUP, resp).Success());
    EXPECT_EQ(flag, (std::vector<std::uint8_t>{0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12,
                                               static_cast<std::uint8_t>(ResponseStatus::Ready),
                                               0x8D, 0x01}));

    KvResponse unpacked;
    ASSERT_TRUE(mgr_.UnpackResponse(flag.data(), OpType::LOOKUP, kRequestId,
                                    static_cast<std::uint16_t>(resp.results.size()), unpacked)
                    .Success());
    EXPECT_EQ(unpacked.results, resp.results);
}

TEST_F(UCDramKvProtocolTest, FourBitResponseCoversEveryPackedByteValue)
{
    for (std::uint16_t low = 0; low <= 0x0FU; ++low) {
        for (std::uint16_t high = 0; high <= 0x0FU; ++high) {
            KvResponse response;
            response.request_id = kRequestId;
            response.results = {static_cast<std::uint8_t>(low), static_cast<std::uint8_t>(high)};
            std::vector<std::uint8_t> flag(
                mgr_.GetPackedResponseSize(OpType::DUMP, response.results.size()), 0xFFU);

            ASSERT_TRUE(mgr_.PackResponse(flag.data(), OpType::DUMP, response).Success());
            ASSERT_EQ(flag.size(), kResponseResultsOffset + 1U);
            EXPECT_EQ(flag[kResponseResultsOffset], static_cast<std::uint8_t>(low | (high << 4U)));

            KvResponse unpacked;
            ASSERT_TRUE(
                mgr_.UnpackResponse(flag.data(), OpType::DUMP, kRequestId, 2U, unpacked).Success());
            EXPECT_EQ(unpacked.results, response.results);
        }
    }
}

TEST_F(UCDramKvProtocolTest, OneBitResponseCoversEveryPackedByteValue)
{
    for (std::uint16_t byteValue = 0; byteValue <= 0xFFU; ++byteValue) {
        KvResponse response;
        response.request_id = kRequestId;
        for (std::size_t bit = 0; bit < 8U; ++bit) {
            response.results.push_back(static_cast<std::uint8_t>((byteValue >> bit) & 0x01U));
        }
        std::vector<std::uint8_t> flag(
            mgr_.GetPackedResponseSize(OpType::LOOKUP, response.results.size()), 0xFFU);

        ASSERT_TRUE(mgr_.PackResponse(flag.data(), OpType::LOOKUP, response).Success());
        ASSERT_EQ(flag.size(), kResponseResultsOffset + 1U);
        EXPECT_EQ(flag[kResponseResultsOffset], static_cast<std::uint8_t>(byteValue));

        KvResponse unpacked;
        ASSERT_TRUE(
            mgr_.UnpackResponse(flag.data(), OpType::LOOKUP, kRequestId, 8U, unpacked).Success());
        EXPECT_EQ(unpacked.results, response.results);
    }
}

TEST_F(UCDramKvProtocolTest, ResponsePackingClearsUnusedBitsAndPreservesCanary)
{
    constexpr std::array<std::size_t, 12> kBoundaryCounts = {1,  2,  3,  7,   8,   9,
                                                             15, 16, 17, 255, 256, 257};
    for (const std::size_t count : kBoundaryCounts) {
        for (const OpType opcode : {OpType::DUMP, OpType::LOAD, OpType::LOOKUP}) {
            KvResponse response;
            response.request_id = kRequestId;
            response.results.resize(count);
            for (std::size_t index = 0; index < count; ++index) {
                response.results[index] = static_cast<std::uint8_t>(
                    opcode == OpType::LOOKUP ? (index + count) & 0x01U
                                             : (index * 7U + count) & 0x0FU);
            }

            const std::size_t packedSize = mgr_.GetPackedResponseSize(opcode, count);
            std::vector<std::uint8_t> flag(packedSize + 8U, 0xA5U);
            ASSERT_TRUE(mgr_.PackResponse(flag.data(), opcode, response).Success());
            for (std::size_t index = packedSize; index < flag.size(); ++index) {
                EXPECT_EQ(flag[index], 0xA5U) << "count=" << count << ", index=" << index;
            }

            if (opcode == OpType::LOOKUP && count % 8U != 0U) {
                EXPECT_EQ(static_cast<unsigned>(flag[packedSize - 1U]) >> (count % 8U), 0U);
            } else if (opcode != OpType::LOOKUP && count % 2U != 0U) {
                EXPECT_EQ(flag[packedSize - 1U] & 0xF0U, 0U);
            }

            KvResponse unpacked;
            ASSERT_TRUE(mgr_.UnpackResponse(flag.data(), opcode, kRequestId,
                                            static_cast<std::uint16_t>(count), unpacked)
                            .Success());
            EXPECT_EQ(unpacked.results, response.results);
        }
    }
}

TEST_F(UCDramKvProtocolTest, PackRequestRejectsNullTarget)
{
    KvLookupRequest request;
    request.opcode = OpType::LOOKUP;
    request.request_id = kRequestId;
    request.resp_addr = 0x1000;
    request.batch_size = 1;
    request.entries = {KvLookupEntry{KeyFromHex("10")}};

    const auto status = mgr_.PackRequest(nullptr, request.opcode, request);

    EXPECT_TRUE(status.Failure());
    EXPECT_NE(status.ToString().find("null"), std::string::npos);
}

TEST_F(UCDramKvProtocolTest, RejectsRequestConcreteTypeMismatch)
{
    KvLoadRequest request;
    request.opcode = OpType::DUMP;
    request.request_id = kRequestId;
    request.resp_addr = 0x1000;
    request.batch_size = 1;
    request.entries = {
        KvLoadEntry{KeyFromHex("10"), 0x2000, 0x100, 0}
    };
    std::array<std::uint8_t, kKvDumpRequestHeaderSize + kKvDumpEntrySize> packed{};

    EXPECT_EQ(mgr_.GetPackedRequestSize(OpType::DUMP, request), 0U);
    const auto status = mgr_.PackRequest(packed.data(), OpType::DUMP, request);
    EXPECT_TRUE(status.Failure());
    EXPECT_NE(status.ToString().find("type mismatch"), std::string::npos);
}

TEST_F(UCDramKvProtocolTest, UnpackRequestRejectsZeroHeaderFieldsForEveryOpcode)
{
    std::vector<std::pair<std::vector<std::uint8_t>, std::size_t>> packedRequests;
    {
        KvDumpRequest request;
        request.opcode = OpType::DUMP;
        request.request_id = kRequestId;
        request.resp_addr = 0x1000;
        request.ttl = 10;
        request.batch_size = 1;
        request.entries = {
            KvDumpEntry{KeyFromHex("10"), 0x2000, 0x100, 0}
        };
        std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(request.opcode, request));
        ASSERT_TRUE(mgr_.PackRequest(packed.data(), request.opcode, request).Success());
        packedRequests.emplace_back(std::move(packed), kDumpBatchSizeOffset);
    }
    {
        KvLoadRequest request;
        request.opcode = OpType::LOAD;
        request.request_id = kRequestId;
        request.resp_addr = 0x1000;
        request.batch_size = 1;
        request.entries = {
            KvLoadEntry{KeyFromHex("20"), 0x2000, 0x100, 0}
        };
        std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(request.opcode, request));
        ASSERT_TRUE(mgr_.PackRequest(packed.data(), request.opcode, request).Success());
        packedRequests.emplace_back(std::move(packed), kLoadLookupBatchSizeOffset);
    }
    {
        KvLookupRequest request;
        request.opcode = OpType::LOOKUP;
        request.request_id = kRequestId;
        request.resp_addr = 0x1000;
        request.batch_size = 1;
        request.entries = {KvLookupEntry{KeyFromHex("30")}};
        std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(request.opcode, request));
        ASSERT_TRUE(mgr_.PackRequest(packed.data(), request.opcode, request).Success());
        packedRequests.emplace_back(std::move(packed), kLoadLookupBatchSizeOffset);
    }

    for (const auto& [packed, batchSizeOffset] : packedRequests) {
        const auto opcode = static_cast<OpType>(packed[kOpcodeOffset]);
        SCOPED_TRACE(static_cast<std::uint8_t>(opcode));
        auto malformed = packed;
        std::fill_n(malformed.begin() + kRequestIdOffset, sizeof(std::uint64_t), std::uint8_t{0});
        std::unique_ptr<KvRequest> output;
        auto status = mgr_.UnpackRequest(malformed.data(), malformed.size(), output);
        EXPECT_TRUE(status.Failure());
        EXPECT_NE(status.ToString().find("request_id"), std::string::npos);

        malformed = packed;
        std::fill_n(malformed.begin() + kRespAddrOffset, sizeof(std::uint64_t), std::uint8_t{0});
        status = mgr_.UnpackRequest(malformed.data(), malformed.size(), output);
        EXPECT_TRUE(status.Failure());
        EXPECT_NE(status.ToString().find("resp_addr"), std::string::npos);

        malformed = packed;
        std::fill_n(malformed.begin() + batchSizeOffset, sizeof(std::uint16_t), std::uint8_t{0});
        status = mgr_.UnpackRequest(malformed.data(), malformed.size(), output);
        EXPECT_TRUE(status.Failure());
        EXPECT_NE(status.ToString().find("batch_size"), std::string::npos);
    }
}

TEST_F(UCDramKvProtocolTest, UnpackRequestRejectsInvalidDumpLoadEntryFields)
{
    auto expectInvalidEntry = [this](const std::vector<std::uint8_t>& packed,
                                     std::size_t headerSize, std::size_t fieldOffset,
                                     std::size_t fieldSize, const char* fieldName) {
        auto malformed = packed;
        std::fill_n(malformed.begin() + headerSize + fieldOffset, fieldSize, std::uint8_t{0});
        std::unique_ptr<KvRequest> output;
        const auto status = mgr_.UnpackRequest(malformed.data(), malformed.size(), output);
        EXPECT_TRUE(status.Failure());
        EXPECT_NE(status.ToString().find(fieldName), std::string::npos);
    };

    KvDumpRequest dump;
    dump.opcode = OpType::DUMP;
    dump.request_id = kRequestId;
    dump.resp_addr = 0x1000;
    dump.ttl = 10;
    dump.batch_size = 1;
    dump.entries = {
        KvDumpEntry{KeyFromHex("10"), 0x2000, 0x100, 0}
    };
    std::vector<std::uint8_t> packedDump(mgr_.GetPackedRequestSize(dump.opcode, dump));
    ASSERT_TRUE(mgr_.PackRequest(packedDump.data(), dump.opcode, dump).Success());

    KvLoadRequest load;
    load.opcode = OpType::LOAD;
    load.request_id = kRequestId;
    load.resp_addr = 0x1000;
    load.batch_size = 1;
    load.entries = {
        KvLoadEntry{KeyFromHex("20"), 0x2000, 0x100, 0}
    };
    std::vector<std::uint8_t> packedLoad(mgr_.GetPackedRequestSize(load.opcode, load));
    ASSERT_TRUE(mgr_.PackRequest(packedLoad.data(), load.opcode, load).Success());

    for (const auto& [packed, headerSize] : {
             std::pair{packedDump, kKvDumpRequestHeaderSize},
             std::pair{packedLoad, kKvLoadRequestHeaderSize}
    }) {
        expectInvalidEntry(packed, headerSize, kDumpLoadEntryKeyOffset, kKvKeySize, "key");
        expectInvalidEntry(packed, headerSize, kDumpLoadEntryAddrOffset, sizeof(std::uint64_t),
                           "addr");
        expectInvalidEntry(packed, headerSize, kDumpLoadEntryLenOffset, sizeof(std::uint32_t),
                           "len");
    }
}

TEST_F(UCDramKvProtocolTest, UnpackRequestFailurePreservesOutput)
{
    std::vector<std::uint8_t> malformed(kKvLookupRequestHeaderSize + kKvLookupEntrySize, 0);
    malformed[kOpcodeOffset] = static_cast<std::uint8_t>(OpType::LOOKUP);
    auto original = std::make_unique<KvLookupRequest>();
    auto* originalAddress = original.get();
    std::unique_ptr<KvRequest> output = std::move(original);

    const auto status = mgr_.UnpackRequest(malformed.data(), malformed.size(), output);

    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(output.get(), originalAddress);
}

}  // namespace
}  // namespace UC::DramPool
