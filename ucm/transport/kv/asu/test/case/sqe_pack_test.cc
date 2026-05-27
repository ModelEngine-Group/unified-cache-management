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
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include "sqe.h"

namespace UC::ASU {
namespace {

class SqePackTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SqePackTest, StoreSqePackMatchesProtocol)
{
    // Test values
    constexpr std::uint32_t kCid = 0x1234;
    constexpr std::uint32_t kKvNsId = 0x0001;
    constexpr std::uint8_t kDtype = 0x1;   // 3 bits [15:13]
    constexpr std::uint8_t kDspec = 0x05;  // 5 bits [12:8]
    constexpr std::uint64_t kBufferAddr = 0x0000123456789ABCULL;
    constexpr std::uint32_t kBufferLength = 0x00010000;  // 64KB, 512B aligned
    constexpr std::uint32_t kMrKey = 0x76543210;
    constexpr std::uint32_t kOffset = 0x00001000;  // 4KB, 512B aligned
    constexpr bool kLr = true;
    constexpr std::uint32_t kLength = 0x00000002;  // 1-based
    const std::string kKey = "test_key_01";

    // Build request
    KvStoreRequest req;
    req.cid = kCid;
    req.kv_ns_id = kKvNsId;
    req.dtype = kDtype;
    req.dspec = kDspec;
    req.buffer_addr = kBufferAddr;
    req.buffer_length = kBufferLength;
    req.mr_key = kMrKey;
    req.offset = kOffset;
    req.lr = kLr;
    req.length = kLength;
    req.key = kKey;

    // Pack
    KvStoreSqe sqe;
    auto status = sqe.Pack(req);
    ASSERT_TRUE(status.ok()) << status.message;

    // Build expected buffer according to protocol spec
    std::vector<std::uint32_t> expected(16, 0);

    // Dword 0: CID[31:16] | Fixed[15:14]=0b11 | Reserved[13:8] | Opcode[7:0]=0x01
    expected[0] = (kCid << 16) | (0x3 << 14) | 0x01;

    // Dword 1: kv_ns_id[31:0]
    expected[1] = kKvNsId;

    // Dword 2: DTYPE[15:13] | DSPEC[12:8] | Reserved[7:0]
    expected[2] = ((kDtype & 0x7) << 13) | ((kDspec & 0x1F) << 8);

    // Dword 3-5: reserved (zero)
    expected[3] = 0;
    expected[4] = 0;
    expected[5] = 0;

    // Dword 6-7: DPTR.buffer[63:0]
    expected[6] = kBufferAddr & 0xFFFFFFFFULL;
    expected[7] = (kBufferAddr >> 32) & 0xFFFFFFFFULL;

    // Dword 8: DPTR.key[0]=MR_KEY[31:24] | DPTR.length[23:0]
    expected[8] = ((kMrKey & 0xFF) << 24) | (kBufferLength & 0xFFFFFF);

    // Dword 9: DPTR.Type[31:24]=0x40 | DPTR.key[3:1][23:0]
    expected[9] = (0x40 << 24) | ((kMrKey >> 8) & 0xFFFFFF);

    // Dword 10: offset[31:0]
    expected[10] = kOffset;

    // Dword 11: LR[31] | Reserved[30:24] | Length[23:0]
    expected[11] = (kLr ? (1U << 31) : 0) | (kLength & 0xFFFFFF);

    // Dword 12-15: key[bytes 15:0] - 16 bytes, low-byte aligned
    std::size_t key_len = std::min(kKey.size(), static_cast<std::size_t>(16));
    if (key_len > 0) { std::memcpy(&expected[12], kKey.data(), key_len); }

    // Compare
    ASSERT_EQ(sqe.Size(), expected.size() * sizeof(std::uint32_t));
    const auto* packed = static_cast<const std::uint32_t*>(sqe.Data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i << ": expected 0x"
                                          << std::hex << expected[i] << ", got 0x" << packed[i];
    }
}

TEST_F(SqePackTest, RetrieveSqePackMatchesProtocol)
{
    constexpr std::uint32_t kCid = 0x5678;
    constexpr std::uint32_t kKvNsId = 0x0002;
    constexpr std::uint64_t kBufferAddr = 0x0000FEDCBA987654ULL;
    constexpr std::uint32_t kBufferLength = 0x00020000;  // 128KB
    constexpr std::uint32_t kMrKey = 0x12345678;
    constexpr std::uint32_t kOffset = 0x00002000;  // 8KB
    constexpr bool kLr = false;
    constexpr std::uint32_t kLength = 0x00000003;
    const std::string kKey = "retrieve_key";

    KvRetrieveRequest req;
    req.cid = kCid;
    req.kv_ns_id = kKvNsId;
    req.buffer_addr = kBufferAddr;
    req.buffer_length = kBufferLength;
    req.mr_key = kMrKey;
    req.offset = kOffset;
    req.lr = kLr;
    req.length = kLength;
    req.key = kKey;

    KvRetrieveSqe sqe;
    auto status = sqe.Pack(req);
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint32_t> expected(16, 0);
    expected[0] = (kCid << 16) | (0x3 << 14) | 0x02;  // Opcode=0x02
    expected[1] = kKvNsId;
    expected[6] = kBufferAddr & 0xFFFFFFFFULL;
    expected[7] = (kBufferAddr >> 32) & 0xFFFFFFFFULL;
    expected[8] = ((kMrKey & 0xFF) << 24) | (kBufferLength & 0xFFFFFF);
    expected[9] = (0x40 << 24) | ((kMrKey >> 8) & 0xFFFFFF);
    expected[10] = kOffset;
    expected[11] = (kLr ? (1U << 31) : 0) | (kLength & 0xFFFFFF);
    std::memcpy(&expected[12], kKey.data(), std::min(kKey.size(), static_cast<std::size_t>(16)));

    ASSERT_EQ(sqe.Size(), expected.size() * sizeof(std::uint32_t));
    const auto* packed = static_cast<const std::uint32_t*>(sqe.Data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i;
    }
}

TEST_F(SqePackTest, BatchStoreSqePackMatchesProtocol)
{
    constexpr std::uint32_t kCid = 0xABCD;
    constexpr std::uint32_t kKvNsId = 0x0003;
    constexpr std::uint8_t kDtype = 0x2;
    constexpr std::uint8_t kDspec = 0x0A;
    constexpr std::uint64_t kRespBufferAddr = 0x0000111122223333ULL;
    constexpr std::uint32_t kRespMrKey = 0x99999999;
    constexpr bool kLr = true;
    constexpr bool kRflag = true;

    KvBatchStoreEntry entry1;
    entry1.offset = 0x1000;
    entry1.key = "batch_key_1";
    entry1.buffer_addr = 0x0000AAAABBBBCCCCULL;
    entry1.mr_key = 0x11111111;
    entry1.length = 0x2000;

    KvBatchStoreEntry entry2;
    entry2.offset = 0x2000;
    entry2.key = "batch_key_2";
    entry2.buffer_addr = 0x0000DDDDEEEEFFFFULL;
    entry2.mr_key = 0x22222222;
    entry2.length = 0x3000;

    KvBatchStoreRequest req;
    req.cid = kCid;
    req.kv_ns_id = kKvNsId;
    req.dtype = kDtype;
    req.dspec = kDspec;
    req.response_buffer_addr = kRespBufferAddr;
    req.response_mr_key = kRespMrKey;
    req.lr = kLr;
    req.rflag = kRflag;
    req.batch_number = 2;
    req.entries = {entry1, entry2};

    KvBatchStoreSqe sqe;
    auto status = sqe.Pack(req);
    ASSERT_TRUE(status.ok()) << status.message;

    // 16 base dwords + 2 entries * 9 dwords = 34 dwords
    std::vector<std::uint32_t> expected(34, 0);
    expected[0] = (kCid << 16) | (0x3 << 14) | (kRflag ? (1U << 13) : 0) | 0x45;
    expected[1] = kKvNsId;
    expected[2] = ((kDtype & 0x7) << 13) | ((kDspec & 0x1F) << 8);
    expected[3] = kRespBufferAddr & 0xFFFFFFFFULL;
    expected[4] = (kRespBufferAddr >> 32) & 0xFFFFFFFFULL;
    expected[5] = kRespMrKey;
    expected[8] = 2 * 36;      // batch_number * 36
    expected[9] = 0x01 << 24;  // DPTR.Type=0x01
    expected[10] = 2;          // batch_number
    expected[11] = kLr ? (1U << 31) : 0;

    // Entry 0 (base=16)
    expected[16] = entry1.offset;
    std::memcpy(&expected[17], entry1.key.data(),
                std::min(entry1.key.size(), static_cast<std::size_t>(16)));
    expected[21] = entry1.buffer_addr & 0xFFFFFFFFULL;
    expected[22] = (entry1.buffer_addr >> 32) & 0xFFFFFFFFULL;
    expected[23] = ((entry1.mr_key & 0xFF) << 24) | (entry1.length & 0xFFFFFF);
    expected[24] = (0x40 << 24) | ((entry1.mr_key >> 8) & 0xFFFFFF);

    // Entry 1 (base=25)
    expected[25] = entry2.offset;
    std::memcpy(&expected[26], entry2.key.data(),
                std::min(entry2.key.size(), static_cast<std::size_t>(16)));
    expected[30] = entry2.buffer_addr & 0xFFFFFFFFULL;
    expected[31] = (entry2.buffer_addr >> 32) & 0xFFFFFFFFULL;
    expected[32] = ((entry2.mr_key & 0xFF) << 24) | (entry2.length & 0xFFFFFF);
    expected[33] = (0x40 << 24) | ((entry2.mr_key >> 8) & 0xFFFFFF);

    ASSERT_EQ(sqe.Size(), expected.size() * sizeof(std::uint32_t));
    const auto* packed = static_cast<const std::uint32_t*>(sqe.Data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i;
    }
}

TEST_F(SqePackTest, BatchRetrieveSqePackMatchesProtocol)
{
    constexpr std::uint32_t kCid = 0x1111;
    constexpr std::uint32_t kKvNsId = 0x0004;
    constexpr std::uint64_t kRespBufferAddr = 0x0000444455556666ULL;
    constexpr std::uint32_t kRespMrKey = 0x88888888;
    constexpr bool kLr = false;
    constexpr bool kRflag = false;

    KvBatchRetrieveEntry entry;
    entry.offset = 0x3000;
    entry.key = "batch_retrieve_key";
    entry.buffer_addr = 0x0000777788889999ULL;
    entry.mr_key = 0x33333333;
    entry.length = 0x4000;

    KvBatchRetrieveRequest req;
    req.cid = kCid;
    req.kv_ns_id = kKvNsId;
    req.response_buffer_addr = kRespBufferAddr;
    req.response_mr_key = kRespMrKey;
    req.lr = kLr;
    req.rflag = kRflag;
    req.batch_number = 1;
    req.entries = {entry};

    KvBatchRetrieveSqe sqe;
    auto status = sqe.Pack(req);
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint32_t> expected(25, 0);       // 16 + 1*9
    expected[0] = (kCid << 16) | (0x3 << 14) | 0x46;  // Opcode=0x46
    expected[1] = kKvNsId;
    expected[3] = kRespBufferAddr & 0xFFFFFFFFULL;
    expected[4] = (kRespBufferAddr >> 32) & 0xFFFFFFFFULL;
    expected[5] = kRespMrKey;
    expected[8] = 1 * 36;
    expected[9] = 0x01 << 24;
    expected[10] = 1;
    expected[16] = entry.offset;
    std::memcpy(&expected[17], entry.key.data(),
                std::min(entry.key.size(), static_cast<std::size_t>(16)));
    expected[21] = entry.buffer_addr & 0xFFFFFFFFULL;
    expected[22] = (entry.buffer_addr >> 32) & 0xFFFFFFFFULL;
    expected[23] = ((entry.mr_key & 0xFF) << 24) | (entry.length & 0xFFFFFF);
    expected[24] = (0x40 << 24) | ((entry.mr_key >> 8) & 0xFFFFFF);

    ASSERT_EQ(sqe.Size(), expected.size() * sizeof(std::uint32_t));
    const auto* packed = static_cast<const std::uint32_t*>(sqe.Data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i;
    }
}

TEST_F(SqePackTest, DeleteSqePackMatchesProtocol)
{
    constexpr std::uint32_t kCid = 0x2222;
    constexpr std::uint32_t kKvNsId = 0x0005;
    constexpr std::uint64_t kRespBufferAddr = 0x0000AAAA0000BBBBULL;
    constexpr std::uint32_t kRespMrKey = 0x77777777;
    constexpr bool kRflag = true;

    KvDeleteRequest req;
    req.cid = kCid;
    req.kv_ns_id = kKvNsId;
    req.response_buffer_addr = kRespBufferAddr;
    req.response_mr_key = kRespMrKey;
    req.rflag = kRflag;
    req.batch_number = 2;
    req.keys = {"delete_key_1", "delete_key_2"};

    KvDeleteSqe sqe;
    auto status = sqe.Pack(req);
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint32_t> expected(24, 0);  // 16 + 2*4
    expected[0] = (kCid << 16) | (0x3 << 14) | (kRflag ? (1U << 13) : 0) | 0x08;
    expected[1] = kKvNsId;
    expected[3] = kRespBufferAddr & 0xFFFFFFFFULL;
    expected[4] = (kRespBufferAddr >> 32) & 0xFFFFFFFFULL;
    expected[5] = kRespMrKey;
    expected[8] = 2 * 16;  // batch_number * 16
    expected[9] = 0x01 << 24;
    expected[10] = 2;
    std::memcpy(&expected[16], req.keys[0].data(),
                std::min(req.keys[0].size(), static_cast<std::size_t>(16)));
    std::memcpy(&expected[20], req.keys[1].data(),
                std::min(req.keys[1].size(), static_cast<std::size_t>(16)));

    ASSERT_EQ(sqe.Size(), expected.size() * sizeof(std::uint32_t));
    const auto* packed = static_cast<const std::uint32_t*>(sqe.Data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i;
    }
}

TEST_F(SqePackTest, ExistSqePackMatchesProtocol)
{
    constexpr std::uint32_t kCid = 0x3333;
    constexpr std::uint32_t kKvNsId = 0x0006;
    constexpr std::uint64_t kRespBufferAddr = 0x0000CCCC0000DDDDULL;
    constexpr std::uint32_t kRespMrKey = 0x66666666;
    constexpr bool kRflag = false;
    constexpr bool kSc = true;

    KvExistRequest req;
    req.cid = kCid;
    req.kv_ns_id = kKvNsId;
    req.response_buffer_addr = kRespBufferAddr;
    req.response_mr_key = kRespMrKey;
    req.rflag = kRflag;
    req.sc = kSc;
    req.batch_number = 1;
    req.keys = {"exist_key"};

    KvExistSqe sqe;
    auto status = sqe.Pack(req);
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint32_t> expected(20, 0);       // 16 + 1*4
    expected[0] = (kCid << 16) | (0x3 << 14) | 0x0C;  // Opcode=0x0C
    expected[1] = kKvNsId;
    expected[3] = kRespBufferAddr & 0xFFFFFFFFULL;
    expected[4] = (kRespBufferAddr >> 32) & 0xFFFFFFFFULL;
    expected[5] = kRespMrKey;
    expected[8] = 1 * 16;
    expected[9] = 0x01 << 24;
    expected[10] = 1 | (kSc ? (1U << 16) : 0);  // batch_number | SC
    std::memcpy(&expected[16], req.keys[0].data(),
                std::min(req.keys[0].size(), static_cast<std::size_t>(16)));

    ASSERT_EQ(sqe.Size(), expected.size() * sizeof(std::uint32_t));
    const auto* packed = static_cast<const std::uint32_t*>(sqe.Data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i;
    }
}

TEST_F(SqePackTest, KeepAliveSqePackMatchesProtocol)
{
    constexpr std::uint32_t kCid = 0x4444;
    constexpr std::uint64_t kRespBufferAddr = 0x0000EEEE0000FFFFULL;
    constexpr std::uint32_t kRespMrKey = 0x55555555;
    constexpr bool kRflag = true;

    KvKeepAliveRequest req;
    req.cid = kCid;
    req.response_buffer_addr = kRespBufferAddr;
    req.response_mr_key = kRespMrKey;
    req.rflag = kRflag;

    KvKeepAliveSqe sqe;
    auto status = sqe.Pack(req);
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint32_t> expected(16, 0);
    expected[0] = (kCid << 16) | (kRflag ? (1U << 13) : 0) | 0xF4;  // Opcode=0xF4, no Fixed bits
    expected[3] = kRespBufferAddr & 0xFFFFFFFFULL;
    expected[4] = (kRespBufferAddr >> 32) & 0xFFFFFFFFULL;
    expected[5] = kRespMrKey;

    ASSERT_EQ(sqe.Size(), expected.size() * sizeof(std::uint32_t));
    const auto* packed = static_cast<const std::uint32_t*>(sqe.Data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i;
    }
}

}  // namespace
}  // namespace UC::ASU
