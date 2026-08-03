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
#include <string>
#include <utility>
#include <vector>
#include "dram/dram_test_common.h"

namespace UC::DramPool {
namespace {

using UC::Test::Dram::KeyFromHex;

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

class KvProtocolTest : public ::testing::Test {
protected:
    ProtocolManager mgr_;
};

TEST_F(KvProtocolTest, PackDumpRequestMatchesLayout)
{
    KvDumpEntry entry;
    entry.key = KeyFromHex("10");
    entry.addr = 0x1122334455667788ULL;
    entry.len = 0xAABBCCDD;
    entry.idx = 0x12345678;

    KvDumpRequest req;
    req.opcode = KvOpcode::Dump;
    req.resp_addr = 0x0102030405060708ULL;
    req.batch_size = 1;
    req.ttl = 0x99AABBCCU;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    ASSERT_TRUE(status.Success()) << status.ToString();

    EXPECT_EQ(packed[0], static_cast<std::uint8_t>(KvOpcode::Dump));
    ExpectLe64(packed, 1, req.resp_addr);
    ExpectLe32(packed, 9, req.ttl);
    ExpectLe16(packed, 13, req.batch_size);
    EXPECT_EQ(packed.size(), kKvDumpRequestHeaderSize + kKvDumpEntrySize);
    EXPECT_EQ(std::memcmp(packed.data() + kKvDumpRequestHeaderSize, entry.key.data(), kKvKeySize),
              0);
    ExpectLe64(packed, kKvDumpRequestHeaderSize + 16, entry.addr);
    ExpectLe32(packed, kKvDumpRequestHeaderSize + 24, entry.len);
    ExpectLe32(packed, kKvDumpRequestHeaderSize + 28, entry.idx);
}

TEST_F(KvProtocolTest, PackLoadRequestMatchesLayout)
{
    KvLoadEntry entry;
    entry.key = KeyFromHex("20");
    entry.addr = 0x8877665544332211ULL;
    entry.len = 0x1000;
    entry.idx = 7;

    KvLoadRequest req;
    req.opcode = KvOpcode::Load;
    req.resp_addr = 0x1111222233334444ULL;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    ASSERT_TRUE(status.Success()) << status.ToString();

    EXPECT_EQ(packed[0], static_cast<std::uint8_t>(KvOpcode::Load));
    ExpectLe64(packed, 1, req.resp_addr);
    ExpectLe16(packed, 9, req.batch_size);
    EXPECT_EQ(std::memcmp(packed.data() + kKvLoadRequestHeaderSize, entry.key.data(), kKvKeySize),
              0);
}

TEST_F(KvProtocolTest, PackLookupRequestMatchesLayout)
{
    KvLookupEntry entry0;
    entry0.key = KeyFromHex("30");
    KvLookupEntry entry1;
    entry1.key = KeyFromHex("40");

    KvLookupRequest req;
    req.opcode = KvOpcode::Lookup;
    req.resp_addr = 0x1234000056780000ULL;
    req.batch_size = 2;
    req.entries = {entry0, entry1};

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    ASSERT_TRUE(status.Success()) << status.ToString();

    EXPECT_EQ(packed[0], static_cast<std::uint8_t>(KvOpcode::Lookup));
    ExpectLe64(packed, 1, req.resp_addr);
    ExpectLe16(packed, 9, req.batch_size);
    EXPECT_EQ(
        std::memcmp(packed.data() + kKvLookupRequestHeaderSize, entry0.key.data(), kKvKeySize), 0);
    EXPECT_EQ(std::memcmp(packed.data() + kKvLookupRequestHeaderSize + kKvLookupEntrySize,
                          entry1.key.data(), kKvKeySize),
              0);
}

TEST_F(KvProtocolTest, RejectsBatchSizeMismatch)
{
    KvLookupEntry entry;
    entry.key = KeyFromHex("50");

    KvLookupRequest req;
    req.opcode = KvOpcode::Lookup;
    req.resp_addr = 0x1000;
    req.batch_size = 2;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(kKvLookupRequestHeaderSize + 2 * kKvLookupEntrySize, 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("batch_size"), std::string::npos);
}

TEST_F(KvProtocolTest, RejectsAllZeroKey)
{
    KvLookupEntry entry;

    KvLookupRequest req;
    req.opcode = KvOpcode::Lookup;
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("key"), std::string::npos);
}

TEST_F(KvProtocolTest, RejectsZeroDumpLoadAddrAndLen)
{
    KvDumpEntry entry;
    entry.key = KeyFromHex("60");
    entry.addr = 0;
    entry.len = 1;

    KvDumpRequest req;
    req.opcode = KvOpcode::Dump;
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

TEST_F(KvProtocolTest, UnpackRequestRejectsWrongSize)
{
    KvLookupEntry entry;
    entry.key = KeyFromHex("70");

    KvLookupRequest req;
    req.opcode = KvOpcode::Lookup;
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

TEST_F(KvProtocolTest, UnpackResponseReadsPackedResults)
{
    std::uint8_t lookupFlag[] = {static_cast<std::uint8_t>(ResponseStatus::Ready), 0x8D, 0x01};
    KvResponse resp;
    auto status = mgr_.UnpackResponse(lookupFlag, KvOpcode::Lookup, 9, resp);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(resp.results, (std::vector<std::uint8_t>{1, 0, 1, 1, 0, 0, 0, 1, 1}));

    std::uint8_t dumpFlag[] = {static_cast<std::uint8_t>(ResponseStatus::Ready), 0x10, 0xF2, 0x03};
    resp.results.clear();
    status = mgr_.UnpackResponse(dumpFlag, KvOpcode::Dump, 5, resp);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(resp.results, (std::vector<std::uint8_t>{0, 1, 2, 15, 3}));
}

TEST_F(KvProtocolTest, ServerRoundTripDumpLoad)
{
    KvLoadEntry entry;
    entry.key = KeyFromHex("80");
    entry.addr = 0xAABBCCDDEEFF0011ULL;
    entry.len = 0x2000;
    entry.idx = 0x55;

    KvLoadRequest req;
    req.opcode = KvOpcode::Load;
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
    EXPECT_EQ(dl.resp_addr, req.resp_addr);
    EXPECT_EQ(dl.batch_size, req.batch_size);
    EXPECT_EQ(std::memcmp(dl.entries[0].key.data(), entry.key.data(), kKvKeySize), 0);
    EXPECT_EQ(dl.entries[0].addr, entry.addr);
    EXPECT_EQ(dl.entries[0].len, entry.len);
    EXPECT_EQ(dl.entries[0].idx, entry.idx);

    // server packs response, client unpacks it
    KvResponse resp;
    resp.results = {0x0};  // batch_size == 1 errcode
    std::vector<std::uint8_t> flag(mgr_.GetPackedResponseSize(req.opcode, resp.results.size()),
                                   0xFF);
    ASSERT_TRUE(mgr_.PackResponse(flag.data(), req.opcode, resp).Success());
    KvResponse resp2;
    ASSERT_TRUE(mgr_.UnpackResponse(flag.data(), req.opcode, req.batch_size, resp2).Success());
    ASSERT_EQ(resp2.results.size(), 1u);
    EXPECT_EQ(resp2.results[0], 0x0U);
}

TEST_F(KvProtocolTest, ServerRoundTripLookup)
{
    KvLookupEntry e0;
    e0.key = KeyFromHex("90");
    KvLookupEntry e1;
    e1.key = KeyFromHex("a0");
    KvLookupRequest req;
    req.opcode = KvOpcode::Lookup;
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
    EXPECT_EQ(lk.resp_addr, req.resp_addr);
    EXPECT_EQ(lk.batch_size, req.batch_size);
    EXPECT_EQ(std::memcmp(lk.entries[0].key.data(), e0.key.data(), kKvKeySize), 0);
    EXPECT_EQ(std::memcmp(lk.entries[1].key.data(), e1.key.data(), kKvKeySize), 0);

    // server packs one existence bit per key, client unpacks it
    KvResponse resp;
    resp.results = {1, 0};
    std::vector<std::uint8_t> flag(mgr_.GetPackedResponseSize(req.opcode, resp.results.size()),
                                   0xFF);
    ASSERT_TRUE(mgr_.PackResponse(flag.data(), req.opcode, resp).Success());
    KvResponse resp2;
    ASSERT_TRUE(mgr_.UnpackResponse(flag.data(), req.opcode, req.batch_size, resp2).Success());
    EXPECT_EQ(resp2.results, resp.results);
}

// ---------------------------------------------------------------------------
// Boundary values: every field set to max
// ---------------------------------------------------------------------------

TEST_F(KvProtocolTest, DumpLoadMaxFieldValuesRoundTrip)
{
    KvLoadEntry entry;
    entry.key.fill(std::byte{0xFF});
    entry.addr = 0xFFFFFFFFFFFFFFFFULL;
    entry.len = 0xFFFFFFFFU;
    entry.idx = 0xFFFFFFFFU;

    KvLoadRequest req;
    req.opcode = KvOpcode::Load;
    req.resp_addr = 0xFFFFFFFFFFFFFFFFULL;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).Success());

    std::unique_ptr<KvRequest> parsed;
    ASSERT_TRUE(mgr_.UnpackRequest(packed.data(), packed.size(), parsed).Success());
    auto& dl = static_cast<KvLoadRequest&>(*parsed);
    EXPECT_EQ(dl.resp_addr, 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_EQ(dl.entries[0].addr, 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_EQ(dl.entries[0].len, 0xFFFFFFFFU);
    EXPECT_EQ(dl.entries[0].idx, 0xFFFFFFFFU);
    EXPECT_EQ(std::memcmp(dl.entries[0].key.data(), entry.key.data(), kKvKeySize), 0);
}

// ---------------------------------------------------------------------------
// Multi-entry: batch_size > 1, verify every entry survives the round-trip
// ---------------------------------------------------------------------------

TEST_F(KvProtocolTest, DumpLoadMultiEntryRoundTrip)
{
    constexpr std::uint16_t kBatch = 5;
    KvDumpRequest req;
    req.opcode = KvOpcode::Dump;
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

TEST_F(KvProtocolTest, LookupMultiEntryRoundTrip)
{
    constexpr std::uint16_t kBatch = 4;
    KvLookupRequest req;
    req.opcode = KvOpcode::Lookup;
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

TEST_F(KvProtocolTest, RejectsNoneOpcodeOnDumpLoad)
{
    KvDumpRequest req;  // opcode defaults to None
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {
        KvDumpEntry{KeyFromHex("10"), 0x2000, 0x100, 0}
    };

    std::vector<std::uint8_t> buf(mgr_.GetPackedRequestSize(KvOpcode::Dump, req), 0);
    auto status = mgr_.PackRequest(buf.data(), KvOpcode::None, req);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("opcode"), std::string::npos);
}

TEST_F(KvProtocolTest, RejectsOpcodeMismatch)
{
    KvDumpRequest req;
    req.opcode = KvOpcode::Lookup;  // wrong opcode for a Dump request
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {
        KvDumpEntry{KeyFromHex("10"), 0x2000, 0x100, 0}
    };

    std::vector<std::uint8_t> buf(mgr_.GetPackedRequestSize(KvOpcode::Dump, req), 0);
    auto status = mgr_.PackRequest(buf.data(), KvOpcode::Dump, req);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("opcode"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Server UnpackRequest edge cases (validation is now merged into UnpackRequest)
// ---------------------------------------------------------------------------

TEST_F(KvProtocolTest, UnpackRequestRejectsUnknownOpcode)
{
    std::vector<std::uint8_t> buf(kKvLookupRequestHeaderSize + kKvLookupEntrySize, 0);
    buf[0] = 0xEE;  // unknown opcode
    std::unique_ptr<KvRequest> parsed;
    auto status = mgr_.UnpackRequest(buf.data(), buf.size(), parsed);
    EXPECT_FALSE(status.Success());
    EXPECT_NE(status.ToString().find("unknown opcode"), std::string::npos);
}

TEST_F(KvProtocolTest, UnpackRequestRejectsExtraBytes)
{
    KvLookupRequest req;
    req.opcode = KvOpcode::Lookup;
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

TEST_F(KvProtocolTest, UnpackRequestRejectsNull)
{
    std::unique_ptr<KvRequest> out;
    auto status = mgr_.UnpackRequest(nullptr, kKvLoadRequestHeaderSize, out);
    EXPECT_FALSE(status.Success());
}

TEST_F(KvProtocolTest, UnpackRequestRejectsTruncatedHeaderForEveryOpcode)
{
    struct TestCase {
        KvOpcode opcode;
        std::size_t header_size;
    };
    const std::array<TestCase, 3> test_cases = {
        {{KvOpcode::Dump, kKvDumpRequestHeaderSize},
         {KvOpcode::Load, kKvLoadRequestHeaderSize},
         {KvOpcode::Lookup, kKvLookupRequestHeaderSize}}
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

TEST_F(KvProtocolTest, UnpackRequestRejectsSizeMismatch)
{
    KvDumpRequest req;
    req.opcode = KvOpcode::Dump;
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

TEST_F(KvProtocolTest, PackResponseRejectsNullData)
{
    KvResponse resp;
    resp.results = {0x0};
    auto status = mgr_.PackResponse(nullptr, KvOpcode::Dump, resp);
    EXPECT_FALSE(status.Success());
}

TEST_F(KvProtocolTest, PackResponseRejectsValuesThatDoNotFitWireWidth)
{
    std::uint8_t flag[2] = {0};
    KvResponse lookup;
    lookup.results = {2};
    auto status = mgr_.PackResponse(flag, KvOpcode::Lookup, lookup);
    EXPECT_FALSE(status.Success()) << status.ToString();
    EXPECT_EQ(flag[kResponseStatusOffset], static_cast<std::uint8_t>(ResponseStatus::Pending));

    KvResponse dump;
    dump.results = {16};
    status = mgr_.PackResponse(flag, KvOpcode::Dump, dump);
    EXPECT_FALSE(status.Success()) << status.ToString();
    EXPECT_EQ(flag[kResponseStatusOffset], static_cast<std::uint8_t>(ResponseStatus::Pending));
}

TEST_F(KvProtocolTest, PackResponseLookupRejectsZeroCount)
{
    KvResponse resp;  // empty results
    std::uint8_t flag[1] = {0};
    auto status = mgr_.PackResponse(flag, KvOpcode::Lookup, resp);
    EXPECT_FALSE(status.Success());
}

TEST_F(KvProtocolTest, PackResponseDumpLoadZeroErrcodes)
{
    KvResponse resp;  // empty, result_count=0
    std::uint8_t flag[1] = {0xFF};
    auto status = mgr_.PackResponse(flag, KvOpcode::Dump, resp);
    EXPECT_TRUE(status.Success());
    EXPECT_EQ(flag[kResponseStatusOffset], static_cast<std::uint8_t>(ResponseStatus::Ready));
}

// ---------------------------------------------------------------------------
// UnpackResponse edge cases
// ---------------------------------------------------------------------------

TEST_F(KvProtocolTest, UnpackResponseRejectsNullData)
{
    KvResponse resp;
    auto status = mgr_.UnpackResponse(nullptr, KvOpcode::Dump, 1, resp);
    EXPECT_FALSE(status.Success());
}

TEST_F(KvProtocolTest, ReportsPendingAndReadyResponseStatus)
{
    bool ready = true;
    const std::uint8_t pending[] = {static_cast<std::uint8_t>(ResponseStatus::Pending)};
    auto status = mgr_.IsResponseReady(pending, ready);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_FALSE(ready);

    const std::uint8_t completed[] = {static_cast<std::uint8_t>(ResponseStatus::Ready)};
    status = mgr_.IsResponseReady(completed, ready);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_TRUE(ready);
}

TEST_F(KvProtocolTest, ResponseStatusRejectsNullAndUnknownValues)
{
    bool ready = true;
    auto status = mgr_.IsResponseReady(nullptr, ready);
    EXPECT_TRUE(status.Failure());
    EXPECT_FALSE(ready);

    const std::uint8_t invalid[] = {2};
    ready = true;
    status = mgr_.IsResponseReady(invalid, ready);
    EXPECT_TRUE(status.Failure());
    EXPECT_FALSE(ready);
}

TEST_F(KvProtocolTest, UnpackResponseReturnsRetryWithoutChangingResultsWhilePending)
{
    const std::uint8_t pending[] = {static_cast<std::uint8_t>(ResponseStatus::Pending), 0xFF};
    KvResponse response;
    response.results = {7, 8};

    const auto status = mgr_.UnpackResponse(pending, KvOpcode::Dump, 2, response);

    EXPECT_EQ(status, Status::Retry());
    EXPECT_EQ(response.results, (std::vector<std::uint8_t>{7, 8}));
}

TEST_F(KvProtocolTest, PackedResponseSizesRoundUpAtBitBoundaries)
{
    EXPECT_EQ(mgr_.GetPackedResponseSize(KvOpcode::Lookup, 1), 2U);
    EXPECT_EQ(mgr_.GetPackedResponseSize(KvOpcode::Lookup, 8), 2U);
    EXPECT_EQ(mgr_.GetPackedResponseSize(KvOpcode::Lookup, 9), 3U);
    EXPECT_EQ(mgr_.GetPackedResponseSize(KvOpcode::Dump, 1), 2U);
    EXPECT_EQ(mgr_.GetPackedResponseSize(KvOpcode::Dump, 2), 2U);
    EXPECT_EQ(mgr_.GetPackedResponseSize(KvOpcode::Load, 3), 3U);
    EXPECT_EQ(mgr_.GetPackedResponseSize(KvOpcode::None, 3), 0U);
}

// ---------------------------------------------------------------------------
// Full response symmetry: PackResponse -> UnpackResponse exact match
// ---------------------------------------------------------------------------

TEST_F(KvProtocolTest, ResponseSymmetryMultipleErrcodes)
{
    KvResponse resp;
    resp.results = {0x0, 0x1, 0x2, 0xE, 0xF};
    constexpr std::uint16_t kCount = 5;
    std::vector<std::uint8_t> flag(mgr_.GetPackedResponseSize(KvOpcode::Dump, kCount), 0xFF);

    ASSERT_TRUE(mgr_.PackResponse(flag.data(), KvOpcode::Dump, resp).Success());
    EXPECT_EQ(flag, (std::vector<std::uint8_t>{static_cast<std::uint8_t>(ResponseStatus::Ready),
                                               0x10, 0xE2, 0x0F}));
    KvResponse resp2;
    ASSERT_TRUE(mgr_.UnpackResponse(flag.data(), KvOpcode::Dump, kCount, resp2).Success());
    ASSERT_EQ(resp2.results.size(), kCount);
    for (std::uint16_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(resp2.results[i], resp.results[i]) << "result " << i;
    }
}

// ---------------------------------------------------------------------------
// Multi-round: same manager handles many sequential operations
// ---------------------------------------------------------------------------

TEST_F(KvProtocolTest, MultiRoundSequentialPacks)
{
    for (std::uint8_t round = 0; round < 10; ++round) {
        const auto opcode = (round % 2 == 0) ? KvOpcode::Dump : KvOpcode::Load;
        const auto resp_addr = 0x1000ULL * (round + 1);
        const auto addr = 0x2000ULL * (round + 1);
        const auto len = 0x100U * (round + 1);

        std::vector<std::uint8_t> packed;
        if (opcode == KvOpcode::Dump) {
            KvDumpRequest req;
            req.opcode = opcode;
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
        if (opcode == KvOpcode::Dump) {
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
        resp.results = {round};
        std::uint8_t flag[2] = {0xFF, 0xFF};
        ASSERT_TRUE(mgr_.PackResponse(flag, opcode, resp).Success()) << "round " << round;
        KvResponse resp2;
        ASSERT_TRUE(mgr_.UnpackResponse(flag, opcode, 1, resp2).Success()) << "round " << round;
        EXPECT_EQ(resp2.results[0], round) << "round " << round;
    }
}

// ---------------------------------------------------------------------------
// Full client-server round-trip with response values spanning the range
// ---------------------------------------------------------------------------

TEST_F(KvProtocolTest, LookupPackedResponseOverwritesNonZeroBufferAndRoundTrips)
{
    KvResponse resp;
    resp.results = {1, 0, 1, 1, 0, 0, 0, 1, 1};
    std::vector<std::uint8_t> flag(
        mgr_.GetPackedResponseSize(KvOpcode::Lookup, resp.results.size()), 0xFF);

    ASSERT_TRUE(mgr_.PackResponse(flag.data(), KvOpcode::Lookup, resp).Success());
    EXPECT_EQ(flag, (std::vector<std::uint8_t>{static_cast<std::uint8_t>(ResponseStatus::Ready),
                                               0x8D, 0x01}));

    KvResponse unpacked;
    ASSERT_TRUE(mgr_.UnpackResponse(flag.data(), KvOpcode::Lookup,
                                    static_cast<std::uint16_t>(resp.results.size()), unpacked)
                    .Success());
    EXPECT_EQ(unpacked.results, resp.results);
}

TEST_F(KvProtocolTest, FourBitResponseCoversEveryPackedByteValue)
{
    for (std::uint16_t low = 0; low <= 0x0FU; ++low) {
        for (std::uint16_t high = 0; high <= 0x0FU; ++high) {
            KvResponse response;
            response.results = {static_cast<std::uint8_t>(low), static_cast<std::uint8_t>(high)};
            std::vector<std::uint8_t> flag(
                mgr_.GetPackedResponseSize(KvOpcode::Dump, response.results.size()), 0xFFU);

            ASSERT_TRUE(mgr_.PackResponse(flag.data(), KvOpcode::Dump, response).Success());
            ASSERT_EQ(flag.size(), 2U);
            EXPECT_EQ(flag[1], static_cast<std::uint8_t>(low | (high << 4U)));

            KvResponse unpacked;
            ASSERT_TRUE(mgr_.UnpackResponse(flag.data(), KvOpcode::Dump, 2U, unpacked).Success());
            EXPECT_EQ(unpacked.results, response.results);
        }
    }
}

TEST_F(KvProtocolTest, OneBitResponseCoversEveryPackedByteValue)
{
    for (std::uint16_t byteValue = 0; byteValue <= 0xFFU; ++byteValue) {
        KvResponse response;
        for (std::size_t bit = 0; bit < 8U; ++bit) {
            response.results.push_back(static_cast<std::uint8_t>((byteValue >> bit) & 0x01U));
        }
        std::vector<std::uint8_t> flag(
            mgr_.GetPackedResponseSize(KvOpcode::Lookup, response.results.size()), 0xFFU);

        ASSERT_TRUE(mgr_.PackResponse(flag.data(), KvOpcode::Lookup, response).Success());
        ASSERT_EQ(flag.size(), 2U);
        EXPECT_EQ(flag[1], static_cast<std::uint8_t>(byteValue));

        KvResponse unpacked;
        ASSERT_TRUE(mgr_.UnpackResponse(flag.data(), KvOpcode::Lookup, 8U, unpacked).Success());
        EXPECT_EQ(unpacked.results, response.results);
    }
}

TEST_F(KvProtocolTest, ResponsePackingClearsUnusedBitsAndPreservesCanary)
{
    constexpr std::array<std::size_t, 12> kBoundaryCounts = {1,  2,  3,  7,   8,   9,
                                                             15, 16, 17, 255, 256, 257};
    for (const std::size_t count : kBoundaryCounts) {
        for (const KvOpcode opcode : {KvOpcode::Dump, KvOpcode::Load, KvOpcode::Lookup}) {
            KvResponse response;
            response.results.resize(count);
            for (std::size_t index = 0; index < count; ++index) {
                response.results[index] = static_cast<std::uint8_t>(
                    opcode == KvOpcode::Lookup ? (index + count) & 0x01U
                                               : (index * 7U + count) & 0x0FU);
            }

            const std::size_t packedSize = mgr_.GetPackedResponseSize(opcode, count);
            std::vector<std::uint8_t> flag(packedSize + 8U, 0xA5U);
            ASSERT_TRUE(mgr_.PackResponse(flag.data(), opcode, response).Success());
            for (std::size_t index = packedSize; index < flag.size(); ++index) {
                EXPECT_EQ(flag[index], 0xA5U) << "count=" << count << ", index=" << index;
            }

            if (opcode == KvOpcode::Lookup && count % 8U != 0U) {
                EXPECT_EQ(static_cast<unsigned>(flag[packedSize - 1U]) >> (count % 8U), 0U);
            } else if (opcode != KvOpcode::Lookup && count % 2U != 0U) {
                EXPECT_EQ(flag[packedSize - 1U] & 0xF0U, 0U);
            }

            KvResponse unpacked;
            ASSERT_TRUE(mgr_.UnpackResponse(flag.data(), opcode, static_cast<std::uint16_t>(count),
                                            unpacked)
                            .Success());
            EXPECT_EQ(unpacked.results, response.results);
        }
    }
}

TEST_F(KvProtocolTest, PackRequestRejectsNullTarget)
{
    KvLookupRequest request;
    request.opcode = KvOpcode::Lookup;
    request.resp_addr = 0x1000;
    request.batch_size = 1;
    request.entries = {KvLookupEntry{KeyFromHex("10")}};

    const auto status = mgr_.PackRequest(nullptr, request.opcode, request);

    EXPECT_TRUE(status.Failure());
    EXPECT_NE(status.ToString().find("null"), std::string::npos);
}

TEST_F(KvProtocolTest, RejectsRequestConcreteTypeMismatch)
{
    KvLoadRequest request;
    request.opcode = KvOpcode::Dump;
    request.resp_addr = 0x1000;
    request.batch_size = 1;
    request.entries = {
        KvLoadEntry{KeyFromHex("10"), 0x2000, 0x100, 0}
    };
    std::array<std::uint8_t, kKvDumpRequestHeaderSize + kKvDumpEntrySize> packed{};

    EXPECT_EQ(mgr_.GetPackedRequestSize(KvOpcode::Dump, request), 0U);
    const auto status = mgr_.PackRequest(packed.data(), KvOpcode::Dump, request);
    EXPECT_TRUE(status.Failure());
    EXPECT_NE(status.ToString().find("type mismatch"), std::string::npos);
}

TEST_F(KvProtocolTest, UnpackRequestRejectsZeroHeaderFieldsForEveryOpcode)
{
    std::vector<std::pair<std::vector<std::uint8_t>, std::size_t>> packedRequests;
    {
        KvDumpRequest request;
        request.opcode = KvOpcode::Dump;
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
        request.opcode = KvOpcode::Load;
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
        request.opcode = KvOpcode::Lookup;
        request.resp_addr = 0x1000;
        request.batch_size = 1;
        request.entries = {KvLookupEntry{KeyFromHex("30")}};
        std::vector<std::uint8_t> packed(mgr_.GetPackedRequestSize(request.opcode, request));
        ASSERT_TRUE(mgr_.PackRequest(packed.data(), request.opcode, request).Success());
        packedRequests.emplace_back(std::move(packed), kLoadLookupBatchSizeOffset);
    }

    for (const auto& [packed, batchSizeOffset] : packedRequests) {
        const auto opcode = static_cast<KvOpcode>(packed[kOpcodeOffset]);
        SCOPED_TRACE(static_cast<std::uint8_t>(opcode));
        auto malformed = packed;
        std::fill_n(malformed.begin() + kRespAddrOffset, sizeof(std::uint64_t), std::uint8_t{0});
        std::unique_ptr<KvRequest> output;
        auto status = mgr_.UnpackRequest(malformed.data(), malformed.size(), output);
        EXPECT_TRUE(status.Failure());
        EXPECT_NE(status.ToString().find("resp_addr"), std::string::npos);

        malformed = packed;
        std::fill_n(malformed.begin() + batchSizeOffset, sizeof(std::uint16_t), std::uint8_t{0});
        status = mgr_.UnpackRequest(malformed.data(), malformed.size(), output);
        EXPECT_TRUE(status.Failure());
        EXPECT_NE(status.ToString().find("batch_size"), std::string::npos);
    }
}

TEST_F(KvProtocolTest, UnpackRequestRejectsInvalidDumpLoadEntryFields)
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
    dump.opcode = KvOpcode::Dump;
    dump.resp_addr = 0x1000;
    dump.ttl = 10;
    dump.batch_size = 1;
    dump.entries = {
        KvDumpEntry{KeyFromHex("10"), 0x2000, 0x100, 0}
    };
    std::vector<std::uint8_t> packedDump(mgr_.GetPackedRequestSize(dump.opcode, dump));
    ASSERT_TRUE(mgr_.PackRequest(packedDump.data(), dump.opcode, dump).Success());

    KvLoadRequest load;
    load.opcode = KvOpcode::Load;
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

TEST_F(KvProtocolTest, UnpackRequestFailurePreservesOutput)
{
    std::vector<std::uint8_t> malformed(kKvLookupRequestHeaderSize + kKvLookupEntrySize, 0);
    malformed[kOpcodeOffset] = static_cast<std::uint8_t>(KvOpcode::Lookup);
    auto original = std::make_unique<KvLookupRequest>();
    auto* originalAddress = original.get();
    std::unique_ptr<KvRequest> output = std::move(original);

    const auto status = mgr_.UnpackRequest(malformed.data(), malformed.size(), output);

    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(output.get(), originalAddress);
}

}  // namespace
}  // namespace UC::DramPool
