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
#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace UC::DRAMPOOL {
namespace {

std::array<std::uint8_t, kKvKeySize> MakeKey(std::uint8_t seed)
{
    std::array<std::uint8_t, kKvKeySize> key{};
    for (std::size_t i = 0; i < key.size(); ++i) { key[i] = static_cast<std::uint8_t>(seed + i); }
    return key;
}

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
    KvDumpLoadEntry entry;
    entry.key = MakeKey(0x10);
    entry.addr = 0x1122334455667788ULL;
    entry.len = 0xAABBCCDD;
    entry.idx = 0x12345678;

    KvDumpLoadRequest req;
    req.opcode = KvOpcode::Dump;
    req.resp_addr = 0x0102030405060708ULL;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(packed[0], static_cast<std::uint8_t>(KvOpcode::Dump));
    ExpectLe64(packed, 1, req.resp_addr);
    ExpectLe16(packed, 9, req.batch_size);
    EXPECT_EQ(std::memcmp(packed.data() + kKvHeaderSize, entry.key.data(), kKvKeySize), 0);
    ExpectLe64(packed, kKvHeaderSize + 16, entry.addr);
    ExpectLe32(packed, kKvHeaderSize + 24, entry.len);
    ExpectLe32(packed, kKvHeaderSize + 28, entry.idx);
}

TEST_F(KvProtocolTest, PackLoadRequestMatchesLayout)
{
    KvDumpLoadEntry entry;
    entry.key = MakeKey(0x20);
    entry.addr = 0x8877665544332211ULL;
    entry.len = 0x1000;
    entry.idx = 7;

    KvDumpLoadRequest req;
    req.opcode = KvOpcode::Load;
    req.resp_addr = 0x1111222233334444ULL;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(packed[0], static_cast<std::uint8_t>(KvOpcode::Load));
    ExpectLe64(packed, 1, req.resp_addr);
    ExpectLe16(packed, 9, req.batch_size);
    EXPECT_EQ(std::memcmp(packed.data() + kKvHeaderSize, entry.key.data(), kKvKeySize), 0);
}

TEST_F(KvProtocolTest, PackLookupRequestMatchesLayout)
{
    KvLookupEntry entry0;
    entry0.key = MakeKey(0x30);
    KvLookupEntry entry1;
    entry1.key = MakeKey(0x40);

    KvLookupRequest req;
    req.opcode = KvOpcode::Lookup;
    req.resp_addr = 0x1234000056780000ULL;
    req.batch_size = 2;
    req.entries = {entry0, entry1};

    std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(packed[0], static_cast<std::uint8_t>(KvOpcode::Lookup));
    ExpectLe64(packed, 1, req.resp_addr);
    ExpectLe16(packed, 9, req.batch_size);
    EXPECT_EQ(std::memcmp(packed.data() + kKvHeaderSize, entry0.key.data(), kKvKeySize), 0);
    EXPECT_EQ(std::memcmp(packed.data() + kKvHeaderSize + kKvLookupEntrySize, entry1.key.data(),
                          kKvKeySize),
              0);
}

TEST_F(KvProtocolTest, RejectsBatchSizeMismatch)
{
    KvLookupEntry entry;
    entry.key = MakeKey(0x50);

    KvLookupRequest req;
    req.opcode = KvOpcode::Lookup;
    req.resp_addr = 0x1000;
    req.batch_size = 2;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(kKvHeaderSize + 2 * kKvLookupEntrySize, 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("batch_size"), std::string::npos);
}

TEST_F(KvProtocolTest, RejectsAllZeroKey)
{
    KvLookupEntry entry;

    KvLookupRequest req;
    req.opcode = KvOpcode::Lookup;
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("key"), std::string::npos);
}

TEST_F(KvProtocolTest, RejectsZeroDumpLoadAddrAndLen)
{
    KvDumpLoadEntry entry;
    entry.key = MakeKey(0x60);
    entry.addr = 0;
    entry.len = 1;

    KvDumpLoadRequest req;
    req.opcode = KvOpcode::Dump;
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("addr"), std::string::npos);

    req.entries[0].addr = 0x2000;
    req.entries[0].len = 0;
    status = mgr_.PackRequest(packed.data(), req.opcode, req);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("len"), std::string::npos);
}

TEST_F(KvProtocolTest, UnpackRequestRejectsWrongSize)
{
    KvLookupEntry entry;
    entry.key = MakeKey(0x70);

    KvLookupRequest req;
    req.opcode = KvOpcode::Lookup;
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
    auto status = mgr_.PackRequest(packed.data(), req.opcode, req);
    ASSERT_TRUE(status.ok()) << status.message;

    std::unique_ptr<KvRequest> parsed;
    status = mgr_.UnpackRequest(packed.data(), packed.size() - 1, parsed);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("size"), std::string::npos);
}

TEST_F(KvProtocolTest, UnpackResponseReadsFlagEntry)
{
    // Lookup: single idx
    std::uint8_t flag[kKvFlagEntrySize] = {0x78, 0x56, 0x34, 0x12};
    KvResponse resp;
    auto status = mgr_.UnpackResponse(flag, KvOpcode::Lookup, 1, resp);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(resp.results.size(), 1u);
    EXPECT_EQ(resp.results[0], 0x12345678U);

    // Dump/Load: batch_size errcodes (one per entry)
    std::uint8_t flag2[2 * kKvFlagEntrySize] = {0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00};
    resp.results.clear();
    status = mgr_.UnpackResponse(flag2, KvOpcode::Dump, 2, resp);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(resp.results.size(), 2u);
    EXPECT_EQ(resp.results[0], 0x12345678U);
    EXPECT_EQ(resp.results[1], 0x00000000U);
}

TEST_F(KvProtocolTest, ServerRoundTripDumpLoad)
{
    KvDumpLoadEntry entry;
    entry.key = MakeKey(0x80);
    entry.addr = 0xAABBCCDDEEFF0011ULL;
    entry.len = 0x2000;
    entry.idx = 0x55;

    KvDumpLoadRequest req;
    req.opcode = KvOpcode::Load;
    req.resp_addr = 0x9988776655443322ULL;
    req.batch_size = 1;
    req.entries = {entry};

    // client packs
    std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).ok());

    // server unpacks (validation merged into UnpackRequest)
    std::unique_ptr<KvRequest> parsed;
    ASSERT_TRUE(mgr_.UnpackRequest(packed.data(), packed.size(), parsed).ok());
    auto& dl = static_cast<KvDumpLoadRequest&>(*parsed);
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
    std::uint8_t flag[kKvFlagEntrySize] = {0};
    ASSERT_TRUE(mgr_.PackResponse(flag, req.opcode, resp).ok());
    KvResponse resp2;
    ASSERT_TRUE(mgr_.UnpackResponse(flag, req.opcode, req.batch_size, resp2).ok());
    ASSERT_EQ(resp2.results.size(), 1u);
    EXPECT_EQ(resp2.results[0], 0x0U);
}

TEST_F(KvProtocolTest, ServerRoundTripLookup)
{
    KvLookupEntry e0;
    e0.key = MakeKey(0x90);
    KvLookupEntry e1;
    e1.key = MakeKey(0xA0);
    KvLookupRequest req;
    req.opcode = KvOpcode::Lookup;
    req.resp_addr = 0x0E0E0E0E0E0E0E0EULL;
    req.batch_size = 2;
    req.entries = {e0, e1};

    // client packs
    std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).ok());

    // server verifies + unpacks
    std::unique_ptr<KvRequest> parsed;
    ASSERT_TRUE(mgr_.UnpackRequest(packed.data(), packed.size(), parsed).ok());
    auto& lk = static_cast<KvLookupRequest&>(*parsed);
    EXPECT_EQ(lk.opcode, req.opcode);
    EXPECT_EQ(lk.resp_addr, req.resp_addr);
    EXPECT_EQ(lk.batch_size, req.batch_size);
    EXPECT_EQ(std::memcmp(lk.entries[0].key.data(), e0.key.data(), kKvKeySize), 0);
    EXPECT_EQ(std::memcmp(lk.entries[1].key.data(), e1.key.data(), kKvKeySize), 0);

    // server packs the single idx response, client unpacks it
    KvResponse resp;
    resp.results = {0xCAFEBABEU};
    std::uint8_t flag[kKvFlagEntrySize] = {0};
    ASSERT_TRUE(mgr_.PackResponse(flag, req.opcode, resp).ok());
    KvResponse resp2;
    ASSERT_TRUE(mgr_.UnpackResponse(flag, req.opcode, 1, resp2).ok());
    ASSERT_EQ(resp2.results.size(), 1u);
    EXPECT_EQ(resp2.results[0], 0xCAFEBABEU);
}

// ---------------------------------------------------------------------------
// Boundary values: every field set to max
// ---------------------------------------------------------------------------

TEST_F(KvProtocolTest, DumpLoadMaxFieldValuesRoundTrip)
{
    KvDumpLoadEntry entry;
    entry.key.fill(0xFF);
    entry.addr = 0xFFFFFFFFFFFFFFFFULL;
    entry.len = 0xFFFFFFFFU;
    entry.idx = 0xFFFFFFFFU;

    KvDumpLoadRequest req;
    req.opcode = KvOpcode::Load;
    req.resp_addr = 0xFFFFFFFFFFFFFFFFULL;
    req.batch_size = 1;
    req.entries = {entry};

    std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).ok());

    std::unique_ptr<KvRequest> parsed;
    ASSERT_TRUE(mgr_.UnpackRequest(packed.data(), packed.size(), parsed).ok());
    auto& dl = static_cast<KvDumpLoadRequest&>(*parsed);
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
    KvDumpLoadRequest req;
    req.opcode = KvOpcode::Dump;
    req.resp_addr = 0xA5A5A5A5A5A5A5A5ULL;
    req.batch_size = kBatch;
    for (std::uint16_t i = 0; i < kBatch; ++i) {
        KvDumpLoadEntry e;
        e.key = MakeKey(static_cast<std::uint8_t>(i * 0x10));
        e.addr = 0x1000ULL * (i + 1);
        e.len = 0x200U * (i + 1);
        e.idx = i;
        req.entries.push_back(e);
    }

    std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).ok());

    std::unique_ptr<KvRequest> parsed;
    ASSERT_TRUE(mgr_.UnpackRequest(packed.data(), packed.size(), parsed).ok());
    auto& dl = static_cast<KvDumpLoadRequest&>(*parsed);
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
        e.key = MakeKey(static_cast<std::uint8_t>(0x50 + i));
        req.entries.push_back(e);
    }

    std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).ok());

    std::unique_ptr<KvRequest> parsed;
    ASSERT_TRUE(mgr_.UnpackRequest(packed.data(), packed.size(), parsed).ok());
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
    KvDumpLoadRequest req;  // opcode defaults to None
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {
        KvDumpLoadEntry{MakeKey(0x10), 0x2000, 0x100, 0}
    };

    std::vector<std::uint8_t> buf(mgr_.GetPackedSize(KvOpcode::Dump, req), 0);
    auto status = mgr_.PackRequest(buf.data(), KvOpcode::None, req);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("opcode"), std::string::npos);
}

TEST_F(KvProtocolTest, RejectsOpcodeMismatch)
{
    KvDumpLoadRequest req;
    req.opcode = KvOpcode::Lookup;  // wrong opcode for a DumpLoad request
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {
        KvDumpLoadEntry{MakeKey(0x10), 0x2000, 0x100, 0}
    };

    std::vector<std::uint8_t> buf(mgr_.GetPackedSize(KvOpcode::Dump, req), 0);
    auto status = mgr_.PackRequest(buf.data(), KvOpcode::Dump, req);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("opcode"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Server UnpackRequest edge cases (validation is now merged into UnpackRequest)
// ---------------------------------------------------------------------------

TEST_F(KvProtocolTest, UnpackRequestRejectsUnknownOpcode)
{
    std::vector<std::uint8_t> buf(kKvHeaderSize + kKvLookupEntrySize, 0);
    buf[0] = 0xEE;  // unknown opcode
    std::unique_ptr<KvRequest> parsed;
    auto status = mgr_.UnpackRequest(buf.data(), buf.size(), parsed);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("unknown opcode"), std::string::npos);
}

TEST_F(KvProtocolTest, UnpackRequestRejectsExtraBytes)
{
    KvLookupRequest req;
    req.opcode = KvOpcode::Lookup;
    req.resp_addr = 0x1000;
    req.batch_size = 1;
    req.entries = {KvLookupEntry{MakeKey(0x70)}};

    std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).ok());
    packed.push_back(0x00);  // one extra byte
    std::unique_ptr<KvRequest> parsed;
    auto status = mgr_.UnpackRequest(packed.data(), packed.size(), parsed);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("size"), std::string::npos);
}

TEST_F(KvProtocolTest, UnpackRequestRejectsNull)
{
    std::unique_ptr<KvRequest> out;
    auto status = mgr_.UnpackRequest(nullptr, kKvHeaderSize, out);
    EXPECT_FALSE(status.ok());
}

TEST_F(KvProtocolTest, UnpackRequestRejectsTooSmall)
{
    std::vector<std::uint8_t> buf(kKvHeaderSize - 1, 0);
    buf[0] = static_cast<std::uint8_t>(KvOpcode::Dump);
    std::unique_ptr<KvRequest> out;
    auto status = mgr_.UnpackRequest(buf.data(), buf.size(), out);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("header"), std::string::npos);
}

TEST_F(KvProtocolTest, UnpackRequestRejectsSizeMismatch)
{
    KvDumpLoadRequest req;
    req.opcode = KvOpcode::Dump;
    req.resp_addr = 0x1000;
    req.batch_size = 2;
    req.entries = {
        KvDumpLoadEntry{MakeKey(0x10), 0x2000, 0x100, 0},
        KvDumpLoadEntry{MakeKey(0x20), 0x3000, 0x200, 1}
    };

    std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
    ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).ok());
    std::unique_ptr<KvRequest> out;
    // truncated by 1 byte
    auto status = mgr_.UnpackRequest(packed.data(), packed.size() - 1, out);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("size"), std::string::npos);
}

// ---------------------------------------------------------------------------
// PackResponse edge cases
// ---------------------------------------------------------------------------

TEST_F(KvProtocolTest, PackResponseRejectsNullData)
{
    KvResponse resp;
    resp.results = {0x0};
    auto status = mgr_.PackResponse(nullptr, KvOpcode::Dump, resp);
    EXPECT_FALSE(status.ok());
}

TEST_F(KvProtocolTest, PackResponseLookupRejectsWrongCount)
{
    KvResponse resp;
    resp.results = {0x0, 0x1};  // size 2, but Lookup must return exactly 1
    std::uint8_t flag[2 * kKvFlagEntrySize] = {0};
    auto status = mgr_.PackResponse(flag, KvOpcode::Lookup, resp);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("must be 1"), std::string::npos);
}

TEST_F(KvProtocolTest, PackResponseLookupRejectsZeroCount)
{
    KvResponse resp;  // empty results
    std::uint8_t flag[kKvFlagEntrySize] = {0};
    auto status = mgr_.PackResponse(flag, KvOpcode::Lookup, resp);
    EXPECT_FALSE(status.ok());
}

TEST_F(KvProtocolTest, PackResponseDumpLoadZeroErrcodes)
{
    KvResponse resp;  // empty, result_count=0
    std::uint8_t flag[1] = {0xFF};
    auto status = mgr_.PackResponse(flag, KvOpcode::Dump, resp);
    EXPECT_TRUE(status.ok());  // 0 errcodes is valid (memcpy 0 bytes)
}

// ---------------------------------------------------------------------------
// UnpackResponse edge cases
// ---------------------------------------------------------------------------

TEST_F(KvProtocolTest, UnpackResponseRejectsNullData)
{
    KvResponse resp;
    auto status = mgr_.UnpackResponse(nullptr, KvOpcode::Dump, 1, resp);
    EXPECT_FALSE(status.ok());
}

TEST_F(KvProtocolTest, UnpackResponseLookupRejectsWrongCount)
{
    std::uint8_t flag[2 * kKvFlagEntrySize] = {0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00};
    KvResponse resp;
    auto status = mgr_.UnpackResponse(flag, KvOpcode::Lookup, 2, resp);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("result_count"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Full response symmetry: PackResponse -> UnpackResponse exact match
// ---------------------------------------------------------------------------

TEST_F(KvProtocolTest, ResponseSymmetryMultipleErrcodes)
{
    KvResponse resp;
    resp.results = {0x0, 0x1, 0x2, 0xDEADBEEF, 0xFFFFFFFF};
    constexpr std::uint16_t kCount = 5;
    std::vector<std::uint8_t> flag(kCount * kKvFlagEntrySize, 0);

    ASSERT_TRUE(mgr_.PackResponse(flag.data(), KvOpcode::Dump, resp).ok());
    KvResponse resp2;
    ASSERT_TRUE(mgr_.UnpackResponse(flag.data(), KvOpcode::Dump, kCount, resp2).ok());
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
        KvDumpLoadRequest req;
        req.opcode = (round % 2 == 0) ? KvOpcode::Dump : KvOpcode::Load;
        req.resp_addr = 0x1000ULL * (round + 1);
        req.batch_size = 1;
        req.entries = {
            KvDumpLoadEntry{MakeKey(round), 0x2000ULL * (round + 1), 0x100U * (round + 1), round}
        };

        std::vector<std::uint8_t> packed(mgr_.GetPackedSize(req.opcode, req), 0);
        ASSERT_TRUE(mgr_.PackRequest(packed.data(), req.opcode, req).ok()) << "round " << round;

        std::unique_ptr<KvRequest> parsed;
        ASSERT_TRUE(mgr_.UnpackRequest(packed.data(), packed.size(), parsed).ok())
            << "round " << round;
        auto& dl = static_cast<KvDumpLoadRequest&>(*parsed);
        EXPECT_EQ(dl.opcode, req.opcode) << "round " << round;
        EXPECT_EQ(dl.resp_addr, req.resp_addr) << "round " << round;
        EXPECT_EQ(dl.entries[0].idx, round) << "round " << round;

        // response round-trip each iteration
        KvResponse resp;
        resp.results = {static_cast<std::uint32_t>(round)};
        std::uint8_t flag[kKvFlagEntrySize] = {0};
        ASSERT_TRUE(mgr_.PackResponse(flag, req.opcode, resp).ok()) << "round " << round;
        KvResponse resp2;
        ASSERT_TRUE(mgr_.UnpackResponse(flag, req.opcode, 1, resp2).ok()) << "round " << round;
        EXPECT_EQ(resp2.results[0], static_cast<std::uint32_t>(round)) << "round " << round;
    }
}

// ---------------------------------------------------------------------------
// Full client-server round-trip with response values spanning the range
// ---------------------------------------------------------------------------

TEST_F(KvProtocolTest, LookupIdxFullRangeSymmetry)
{
    const std::uint32_t kValues[] = {0x0, 0x1, 0x7FFFFFFF, 0x80000000, 0xDEADBEEF, 0xFFFFFFFF};
    for (std::uint32_t v : kValues) {
        KvResponse resp;
        resp.results = {v};
        std::uint8_t flag[kKvFlagEntrySize] = {0};
        ASSERT_TRUE(mgr_.PackResponse(flag, KvOpcode::Lookup, resp).ok());
        KvResponse resp2;
        ASSERT_TRUE(mgr_.UnpackResponse(flag, KvOpcode::Lookup, 1, resp2).ok());
        ASSERT_EQ(resp2.results.size(), 1u);
        EXPECT_EQ(resp2.results[0], v);
    }
}

}  // namespace
}  // namespace UC::DRAMPOOL
