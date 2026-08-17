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
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <string_view>

namespace UC::ASU {
namespace {

CacheKey MakeCacheKey(std::string_view text)
{
    CacheKey key{};
    const auto size = std::min(text.size(), key.size());
    if (size > 0) { std::memcpy(key.data(), text.data(), size); }
    return key;
}

CacheKey MakeCacheKey(std::uint64_t value)
{
    CacheKey key{};
    std::memcpy(key.data(), &value, key.size());
    return key;
}

class KvProtocolPackTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(KvProtocolPackTest, StoreProtocolPackMatchesProtocol)
{
    constexpr std::uint16_t kCid = 0x1234;
    constexpr std::uint32_t kKvNsId = 0x0001;
    constexpr std::uint8_t kDtype = 0x1;
    constexpr std::uint8_t kDspec = 0x05;
    constexpr std::uint64_t kBufferAddr = 0x0000123456789ABCULL;
    constexpr std::uint32_t kBufferLength = 0x00010000;
    constexpr std::uint32_t kMrKey = 0x76543210;
    constexpr std::uint32_t kOffset = 0x00001000;
    constexpr bool kLr = true;
    constexpr std::uint32_t kLength = 0x00000002;
    const std::string kKey = "test_key_01";

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
    req.key = MakeCacheKey(kKey);

    KvStoreProtocol proto;
    std::vector<std::uint32_t> packed(16, 0);
    auto status = proto.PackSqe(req, packed.data());
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint32_t> expected(16, 0);
    expected[0] = (kCid << 16) | (0x3 << 14) | 0x01;
    expected[1] = kKvNsId;
    expected[2] = ((kDtype & 0x7) << 13) | ((kDspec & 0x1F) << 8);
    expected[6] = kBufferAddr & 0xFFFFFFFFULL;
    expected[7] = (kBufferAddr >> 32) & 0xFFFFFFFFULL;
    expected[8] = ((kMrKey & 0xFF) << 24) | (kBufferLength & 0xFFFFFF);
    expected[9] = (0x40 << 24) | ((kMrKey >> 8) & 0xFFFFFF);
    expected[10] = kOffset;
    expected[11] = (kLr ? (1U << 31) : 0) | (kLength & 0xFFFFFF);
    std::size_t key_len = std::min(kKey.size(), kCacheKeySizeBytes);
    if (key_len > 0) { std::memcpy(&expected[12], kKey.data(), key_len); }

    ASSERT_EQ(packed.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i << ": expected 0x"
                                          << std::hex << expected[i] << ", got 0x" << packed[i];
    }
}

TEST_F(KvProtocolPackTest, RetrieveProtocolPackMatchesProtocol)
{
    constexpr std::uint16_t kCid = 0x5678;
    constexpr std::uint32_t kKvNsId = 0x0002;
    constexpr std::uint64_t kBufferAddr = 0x0000FEDCBA987654ULL;
    constexpr std::uint32_t kBufferLength = 0x00020000;
    constexpr std::uint32_t kMrKey = 0x12345678;
    constexpr std::uint32_t kOffset = 0x00002000;
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
    req.key = MakeCacheKey(kKey);

    KvRetrieveProtocol proto;
    std::vector<std::uint32_t> packed(proto.PackedSize(req) / sizeof(std::uint32_t), 0);
    auto status = proto.PackSqe(req, packed.data());
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint32_t> expected(16, 0);
    expected[0] = (kCid << 16) | (0x3 << 14) | 0x02;
    expected[1] = kKvNsId;
    expected[6] = kBufferAddr & 0xFFFFFFFFULL;
    expected[7] = (kBufferAddr >> 32) & 0xFFFFFFFFULL;
    expected[8] = ((kMrKey & 0xFF) << 24) | (kBufferLength & 0xFFFFFF);
    expected[9] = (0x40 << 24) | ((kMrKey >> 8) & 0xFFFFFF);
    expected[10] = kOffset;
    expected[11] = (kLr ? (1U << 31) : 0) | (kLength & 0xFFFFFF);
    std::memcpy(&expected[12], kKey.data(), std::min(kKey.size(), kCacheKeySizeBytes));

    ASSERT_EQ(packed.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i;
    }
}

TEST_F(KvProtocolPackTest, BatchStoreProtocolPackMatchesProtocol)
{
    constexpr std::uint16_t kCid = 0xABCD;
    constexpr std::uint32_t kKvNsId = 0x0003;
    constexpr std::uint8_t kDtype = 0x2;
    constexpr std::uint8_t kDspec = 0x0A;
    constexpr std::uint64_t kRespBufferAddr = 0x0000111122223333ULL;
    constexpr std::uint32_t kRespMrKey = 0x99999999;
    constexpr bool kLr = true;
    constexpr bool kRflag = true;

    KvBatchStoreEntry entry1;
    entry1.offset = 0x1000;
    entry1.key = MakeCacheKey("batch_key_1");
    entry1.buffer_addr = 0x0000AAAABBBBCCCCULL;
    entry1.mr_key = 0x11111111;
    entry1.length = 0x2000;

    KvBatchStoreEntry entry2;
    entry2.offset = 0x2000;
    entry2.key = MakeCacheKey("batch_key_2");
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

    KvBatchStoreProtocol proto;
    std::vector<std::uint32_t> packed(proto.PackedSize(req) / sizeof(std::uint32_t), 0);
    auto status = proto.PackSqe(req, packed.data());
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint32_t> expected(34, 0);
    expected[0] = (kCid << 16) | (0x3 << 14) | (kRflag ? (1U << 13) : 0) | 0x45;
    expected[1] = kKvNsId;
    expected[2] = ((kDtype & 0x7) << 13) | ((kDspec & 0x1F) << 8);
    expected[3] = kRespBufferAddr & 0xFFFFFFFFULL;
    expected[4] = (kRespBufferAddr >> 32) & 0xFFFFFFFFULL;
    expected[5] = kRespMrKey;
    expected[8] = 2 * 36;
    expected[9] = 0x01 << 24;
    expected[10] = 2;
    expected[11] = kLr ? (1U << 31) : 0;

    expected[16] = entry1.offset;
    std::memcpy(&expected[17], entry1.key.data(), std::min(entry1.key.size(), kCacheKeySizeBytes));
    expected[21] = entry1.buffer_addr & 0xFFFFFFFFULL;
    expected[22] = (entry1.buffer_addr >> 32) & 0xFFFFFFFFULL;
    expected[23] = ((entry1.mr_key & 0xFF) << 24) | (entry1.length & 0xFFFFFF);
    expected[24] = (0x40 << 24) | ((entry1.mr_key >> 8) & 0xFFFFFF);

    expected[25] = entry2.offset;
    std::memcpy(&expected[26], entry2.key.data(), std::min(entry2.key.size(), kCacheKeySizeBytes));
    expected[30] = entry2.buffer_addr & 0xFFFFFFFFULL;
    expected[31] = (entry2.buffer_addr >> 32) & 0xFFFFFFFFULL;
    expected[32] = ((entry2.mr_key & 0xFF) << 24) | (entry2.length & 0xFFFFFF);
    expected[33] = (0x40 << 24) | ((entry2.mr_key >> 8) & 0xFFFFFF);

    ASSERT_EQ(packed.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i;
    }
}

TEST_F(KvProtocolPackTest, BatchRetrieveProtocolPackMatchesProtocol)
{
    constexpr std::uint16_t kCid = 0x1111;
    constexpr std::uint32_t kKvNsId = 0x0004;
    constexpr std::uint64_t kRespBufferAddr = 0x0000444455556666ULL;
    constexpr std::uint32_t kRespMrKey = 0x88888888;
    constexpr bool kLr = false;
    constexpr bool kRflag = true;

    KvBatchRetrieveEntry entry;
    entry.offset = 0x3000;
    entry.key = MakeCacheKey("batch_ret_key");
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

    KvBatchRetrieveProtocol proto;
    std::vector<std::uint32_t> packed(proto.PackedSize(req) / sizeof(std::uint32_t), 0);
    auto status = proto.PackSqe(req, packed.data());
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint32_t> expected(25, 0);
    expected[0] = (kCid << 16) | (0x3 << 14) | (1U << 13) | 0x46;
    expected[1] = kKvNsId;
    expected[3] = kRespBufferAddr & 0xFFFFFFFFULL;
    expected[4] = (kRespBufferAddr >> 32) & 0xFFFFFFFFULL;
    expected[5] = kRespMrKey;
    expected[8] = 1 * 36;
    expected[9] = 0x01 << 24;
    expected[10] = 1;
    expected[16] = entry.offset;
    std::memcpy(&expected[17], entry.key.data(), std::min(entry.key.size(), kCacheKeySizeBytes));
    expected[21] = entry.buffer_addr & 0xFFFFFFFFULL;
    expected[22] = (entry.buffer_addr >> 32) & 0xFFFFFFFFULL;
    expected[23] = ((entry.mr_key & 0xFF) << 24) | (entry.length & 0xFFFFFF);
    expected[24] = (0x40 << 24) | ((entry.mr_key >> 8) & 0xFFFFFF);

    ASSERT_EQ(packed.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i;
    }
}

TEST_F(KvProtocolPackTest, DeleteProtocolPackMatchesProtocol)
{
    constexpr std::uint16_t kCid = 0x2222;
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
    req.keys = {MakeCacheKey("delete_key_1"), MakeCacheKey("delete_key_2")};

    KvDeleteProtocol proto;
    std::vector<std::uint32_t> packed(proto.PackedSize(req) / sizeof(std::uint32_t), 0);
    auto status = proto.PackSqe(req, packed.data());
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint32_t> expected(24, 0);
    expected[0] = (kCid << 16) | (0x3 << 14) | (kRflag ? (1U << 13) : 0) | 0x08;
    expected[1] = kKvNsId;
    expected[3] = kRespBufferAddr & 0xFFFFFFFFULL;
    expected[4] = (kRespBufferAddr >> 32) & 0xFFFFFFFFULL;
    expected[5] = kRespMrKey;
    expected[8] = 2 * 16;
    expected[9] = 0x01 << 24;
    expected[10] = 2;
    std::memcpy(&expected[16], req.keys[0].data(),
                std::min(req.keys[0].size(), kCacheKeySizeBytes));
    std::memcpy(&expected[20], req.keys[1].data(),
                std::min(req.keys[1].size(), kCacheKeySizeBytes));

    ASSERT_EQ(packed.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i;
    }
}

TEST_F(KvProtocolPackTest, ExistProtocolPackMatchesProtocol)
{
    constexpr std::uint16_t kCid = 0x3333;
    constexpr std::uint32_t kKvNsId = 0x0006;
    constexpr std::uint64_t kRespBufferAddr = 0x0000CCCC0000DDDDULL;
    constexpr std::uint32_t kRespMrKey = 0x66666666;
    constexpr bool kRflag = true;
    constexpr bool kSc = true;

    KvExistRequest req;
    req.cid = kCid;
    req.kv_ns_id = kKvNsId;
    req.response_buffer_addr = kRespBufferAddr;
    req.response_mr_key = kRespMrKey;
    req.rflag = kRflag;
    req.sc = kSc;
    req.batch_number = 1;
    req.keys = {MakeCacheKey("exist_key")};

    KvExistProtocol proto;
    std::vector<std::uint32_t> packed(proto.PackedSize(req) / sizeof(std::uint32_t), 0);
    auto status = proto.PackSqe(req, packed.data());
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint32_t> expected(20, 0);
    expected[0] = (kCid << 16) | (0x3 << 14) | (1U << 13) | 0x0C;
    expected[1] = kKvNsId;
    expected[3] = kRespBufferAddr & 0xFFFFFFFFULL;
    expected[4] = (kRespBufferAddr >> 32) & 0xFFFFFFFFULL;
    expected[5] = kRespMrKey;
    expected[8] = 1 * 16;
    expected[9] = 0x01 << 24;
    expected[10] = 1 | (kSc ? (1U << 16) : 0);
    std::memcpy(&expected[16], req.keys[0].data(),
                std::min(req.keys[0].size(), kCacheKeySizeBytes));

    ASSERT_EQ(packed.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i;
    }
}

TEST_F(KvProtocolPackTest, KeepAliveProtocolPackMatchesProtocol)
{
    constexpr std::uint16_t kCid = 0x4444;
    constexpr std::uint64_t kRespBufferAddr = 0x0000EEEE0000FFFFULL;
    constexpr std::uint32_t kRespMrKey = 0x55555555;
    constexpr bool kRflag = true;

    KvKeepAliveRequest req;
    req.cid = kCid;
    req.response_buffer_addr = kRespBufferAddr;
    req.response_mr_key = kRespMrKey;
    req.rflag = kRflag;

    KvKeepAliveProtocol proto;
    std::vector<std::uint32_t> packed(proto.PackedSize(req) / sizeof(std::uint32_t), 0);
    auto status = proto.PackSqe(req, packed.data());
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint32_t> expected(16, 0);
    expected[0] = (kCid << 16) | (kRflag ? (1U << 13) : 0) | 0xF4;
    expected[3] = kRespBufferAddr & 0xFFFFFFFFULL;
    expected[4] = (kRespBufferAddr >> 32) & 0xFFFFFFFFULL;
    expected[5] = kRespMrKey;

    ASSERT_EQ(packed.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(packed[i], expected[i]) << "Mismatch at Dword " << i;
    }
}

TEST_F(KvProtocolPackTest, StoreAndRetrieveUnpackCqe)
{
    KvStoreProtocol store_proto;
    KvResponse resp;
    std::uint32_t cqe_data[4] = {0, 0, 0, 0};
    cqe_data[3] = 0x1234 | (0 << 17);
    auto status = store_proto.UnpackCqe(cqe_data, 0, resp);
    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(resp.cid, 0x1234);
    EXPECT_EQ(resp.status, 0);

    KvRetrieveProtocol retrieve_proto;
    status = retrieve_proto.UnpackCqe(cqe_data, 0, resp);
    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(resp.cid, 0x1234);
    EXPECT_EQ(resp.status, 0);
}

TEST_F(KvProtocolPackTest, BatchStoreUnpackCqe)
{
    KvResponse resp;
    std::uint32_t cqe_data[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    cqe_data[3] = 0x1234 | (0 << 17);

    KvBatchStoreProtocol proto;
    auto status = proto.UnpackCqe(cqe_data, 2, resp);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(resp.cid, 0x1234);
}

TEST_F(KvProtocolPackTest, ExistUnpackCqe)
{
    KvResponse resp;
    std::uint32_t cqe_data[8] = {0x0005, 0, 0, 0x1234, 0, 0, 0, 0};

    KvExistProtocol proto;
    auto status = proto.UnpackCqe(cqe_data, 3, resp);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(resp.cid, 0x1234);
    EXPECT_EQ(resp.existing_key_number, 5);
}

TEST_F(KvProtocolPackTest, StoreValidateRequestRejectsZeroBufferAddr)
{
    KvStoreRequest req;
    req.buffer_addr = 0;
    req.buffer_length = 512;
    req.offset = 0;
    req.length = 1;
    req.key = MakeCacheKey("key");

    KvStoreProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("buffer_addr is zero"), std::string::npos);
}

TEST_F(KvProtocolPackTest, StoreValidateRequestRejectsZeroBufferLength)
{
    KvStoreRequest req;
    req.buffer_addr = 0x1000;
    req.buffer_length = 0;
    req.offset = 0;
    req.length = 1;
    req.key = MakeCacheKey("key");

    KvStoreProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("buffer_length is zero"), std::string::npos);
}

TEST_F(KvProtocolPackTest, StoreValidateRequestRejectsUnalignedBufferLength)
{
    KvStoreRequest req;
    req.buffer_addr = 0x1000;
    req.buffer_length = 100;
    req.offset = 0;
    req.length = 1;
    req.key = MakeCacheKey("key");

    KvStoreProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("512B aligned"), std::string::npos);
}

TEST_F(KvProtocolPackTest, StoreValidateRequestRejectsAllZeroKey)
{
    KvStoreRequest req;
    req.buffer_addr = 0x1000;
    req.buffer_length = 512;
    req.offset = 0;
    req.length = 1;
    req.key = CacheKey{};

    KvStoreProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("all zeros"), std::string::npos);
}

TEST_F(KvProtocolPackTest, StoreValidateRequestRejectsDtypeOverflow)
{
    KvStoreRequest req;
    req.buffer_addr = 0x1000;
    req.buffer_length = 512;
    req.offset = 0;
    req.length = 1;
    req.key = MakeCacheKey("key");
    req.dtype = 8;

    KvStoreProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("dtype("), std::string::npos);
    EXPECT_NE(status.message.find("exceeds 3-bit limit"), std::string::npos);
}

TEST_F(KvProtocolPackTest, StoreValidateRequestRejectsDspecOverflow)
{
    KvStoreRequest req;
    req.buffer_addr = 0x1000;
    req.buffer_length = 512;
    req.offset = 0;
    req.length = 1;
    req.key = MakeCacheKey("key");
    req.dspec = 32;

    KvStoreProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("dspec("), std::string::npos);
    EXPECT_NE(status.message.find("exceeds 5-bit limit"), std::string::npos);
}

TEST_F(KvProtocolPackTest, StoreValidateRequestAcceptsValidRequest)
{
    KvStoreRequest req;
    req.buffer_addr = 0x1000;
    req.buffer_length = 512;
    req.offset = 0;
    req.length = 1;
    req.key = MakeCacheKey("valid_key");
    req.dtype = 1;
    req.dspec = 5;

    KvStoreProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_TRUE(status.ok()) << status.message;
}

TEST_F(KvProtocolPackTest, BatchStoreValidateRequestRejectsZeroBatchNumber)
{
    KvBatchStoreRequest req;
    req.batch_number = 0;

    KvBatchStoreProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("batch_number("), std::string::npos);
    EXPECT_NE(status.message.find("must be in range"), std::string::npos);
}

TEST_F(KvProtocolPackTest, BatchStoreValidateRequestRejectsRflagWithZeroResponseAddr)
{
    KvBatchStoreRequest req;
    req.batch_number = 1;
    req.rflag = true;
    req.response_buffer_addr = 0;
    req.entries.resize(1);
    req.entries[0].key = MakeCacheKey("key");

    KvBatchStoreProtocol proto;
    std::vector<std::uint32_t> target(64, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("response_buffer_addr is zero"), std::string::npos);
}

TEST_F(KvProtocolPackTest, BatchStoreValidateRequestRejectsUnalignedLength)
{
    KvBatchStoreRequest req;
    req.batch_number = 1;
    req.entries.resize(1);
    req.entries[0].key = MakeCacheKey("key");
    req.entries[0].offset = 0;
    req.entries[0].buffer_addr = 0x1000;
    req.entries[0].length = 100;

    KvBatchStoreProtocol proto;
    std::vector<std::uint32_t> target(64, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("512B aligned"), std::string::npos);
}

TEST_F(KvProtocolPackTest, BatchStoreValidateRequestRejectsAllZeroKey)
{
    KvBatchStoreRequest req;
    req.batch_number = 1;
    req.entries.resize(1);
    req.entries[0].key = CacheKey{};
    req.entries[0].offset = 0;
    req.entries[0].buffer_addr = 0x1000;
    req.entries[0].length = 512;

    KvBatchStoreProtocol proto;
    std::vector<std::uint32_t> target(64, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("all zeros"), std::string::npos);
}

TEST_F(KvProtocolPackTest, KeepAliveValidateRequestRejectsRflagWithZeroResponseAddr)
{
    KvKeepAliveRequest req;
    req.rflag = true;
    req.response_buffer_addr = 0;

    KvKeepAliveProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("response_buffer_addr is zero"), std::string::npos);
}

TEST_F(KvProtocolPackTest, KeepAliveValidateRequestAcceptsNoRflag)
{
    KvKeepAliveRequest req;
    req.rflag = false;
    req.response_buffer_addr = 0;

    KvKeepAliveProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_TRUE(status.ok()) << status.message;
}

TEST_F(KvProtocolPackTest, BatchStoreUnpackCqeResultBuffer)
{
    KvResponse resp;
    std::uint32_t cqe_data[8] = {0};
    cqe_data[3] = 0x5678 | (0x123 << 17);
    cqe_data[4] = 0x0 | (0x1 << 4) | (0x3 << 8);

    KvBatchStoreProtocol proto;
    auto status = proto.UnpackCqe(cqe_data, 3, resp);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(resp.cid, 0x5678);
    EXPECT_EQ(resp.status, 0x123);
    ASSERT_EQ(resp.result_buffer.size(), 3);
    EXPECT_EQ(resp.result_buffer[0], 0x0);
    EXPECT_EQ(resp.result_buffer[1], 0x1);
    EXPECT_EQ(resp.result_buffer[2], 0x3);
}

TEST_F(KvProtocolPackTest, BatchRetrieveUnpackCqe)
{
    KvResponse resp;
    std::uint32_t cqe_data[8] = {0};
    cqe_data[3] = 0x9ABC | (0x456 << 17);
    cqe_data[4] = 0x2 | (0x0 << 4);

    KvBatchRetrieveProtocol proto;
    auto status = proto.UnpackCqe(cqe_data, 2, resp);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(resp.cid, 0x9ABC);
    EXPECT_EQ(resp.status, 0x456);
    ASSERT_EQ(resp.result_buffer.size(), 2);
    EXPECT_EQ(resp.result_buffer[0], 0x2);
    EXPECT_EQ(resp.result_buffer[1], 0x0);
}

TEST_F(KvProtocolPackTest, DeleteUnpackCqe)
{
    KvResponse resp;
    std::uint32_t cqe_data[8] = {0};
    cqe_data[3] = 0xDEF0;
    cqe_data[4] = 0x5;

    KvDeleteProtocol proto;
    auto status = proto.UnpackCqe(cqe_data, 3, resp);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(resp.cid, 0xDEF0);
    ASSERT_EQ(resp.result_buffer.size(), 3);
    EXPECT_EQ(resp.result_buffer[0], 1);
    EXPECT_EQ(resp.result_buffer[1], 0);
    EXPECT_EQ(resp.result_buffer[2], 1);
}

TEST_F(KvProtocolPackTest, ExistUnpackCqeResultBuffer)
{
    KvResponse resp;
    std::uint32_t cqe_data[8] = {0};
    cqe_data[0] = 0x000A;
    cqe_data[3] = 0x1111;
    cqe_data[4] = 0x3;

    KvExistProtocol proto;
    auto status = proto.UnpackCqe(cqe_data, 2, resp);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(resp.cid, 0x1111);
    EXPECT_EQ(resp.existing_key_number, 0x000A);
    ASSERT_EQ(resp.result_buffer.size(), 2);
    EXPECT_EQ(resp.result_buffer[0], 1);
    EXPECT_EQ(resp.result_buffer[1], 1);
}

TEST_F(KvProtocolPackTest, BatchStoreUnpackCqeCrossDword)
{
    KvResponse resp;
    std::uint32_t cqe_data[8] = {0};
    cqe_data[3] = 0x2222;
    cqe_data[4] = 0xFFFFFFFF;
    cqe_data[5] = 0x4;

    KvBatchStoreProtocol proto;
    auto status = proto.UnpackCqe(cqe_data, 9, resp);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(resp.cid, 0x2222);
    ASSERT_EQ(resp.result_buffer.size(), 9);
    for (int i = 0; i < 8; ++i) { EXPECT_EQ(resp.result_buffer[i], 0xF) << "key " << i; }
    EXPECT_EQ(resp.result_buffer[8], 0x4);
}

TEST_F(KvProtocolPackTest, DeleteUnpackCqeSingleKey)
{
    KvResponse resp;
    std::uint32_t cqe_data[8] = {0};
    cqe_data[3] = 0x3333;
    cqe_data[4] = 0x1;

    KvDeleteProtocol proto;
    auto status = proto.UnpackCqe(cqe_data, 1, resp);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(resp.cid, 0x3333);
    ASSERT_EQ(resp.result_buffer.size(), 1);
    EXPECT_EQ(resp.result_buffer[0], 1);
}

TEST_F(KvProtocolPackTest, RetrieveValidateRequestAcceptsValid)
{
    KvRetrieveRequest req;
    req.buffer_addr = 0x2000;
    req.buffer_length = 1024;
    req.offset = 512;
    req.length = 512;
    req.key = MakeCacheKey("retrieve_key");

    KvRetrieveProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_TRUE(status.ok()) << status.message;
}

TEST_F(KvProtocolPackTest, RetrieveValidateRequestRejectsZeroBufferAddr)
{
    KvRetrieveRequest req;
    req.buffer_addr = 0;
    req.buffer_length = 1024;
    req.offset = 0;
    req.length = 1;
    req.key = MakeCacheKey("key");

    KvRetrieveProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("buffer_addr is zero"), std::string::npos);
}

TEST_F(KvProtocolPackTest, RetrieveValidateRequestRejectsUnalignedBufferLength)
{
    KvRetrieveRequest req;
    req.buffer_addr = 0x2000;
    req.buffer_length = 100;
    req.offset = 512;
    req.length = 512;
    req.key = MakeCacheKey("retrieve_key");

    KvRetrieveProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("512B aligned"), std::string::npos);
}

TEST_F(KvProtocolPackTest, RetrieveValidateRequestRejectsAllZeroKey)
{
    KvRetrieveRequest req;
    req.buffer_addr = 0x2000;
    req.buffer_length = 1024;
    req.offset = 512;
    req.length = 512;
    req.key = CacheKey{};

    KvRetrieveProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("all zeros"), std::string::npos);
}

TEST_F(KvProtocolPackTest, BatchRetrieveValidateRequestAcceptsValid)
{
    KvBatchRetrieveRequest req;
    req.batch_number = 2;
    req.entries.resize(2);
    req.entries[0].key = MakeCacheKey("key1");
    req.entries[0].offset = 0;
    req.entries[0].buffer_addr = 0x1000;
    req.entries[0].length = 512;
    req.entries[1].key = MakeCacheKey("key2");
    req.entries[1].offset = 512;
    req.entries[1].buffer_addr = 0x2000;
    req.entries[1].length = 512;

    KvBatchRetrieveProtocol proto;
    std::vector<std::uint32_t> target(64, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_TRUE(status.ok()) << status.message;
}

TEST_F(KvProtocolPackTest, BatchRetrieveValidateRequestRejectsMismatch)
{
    KvBatchRetrieveRequest req;
    req.batch_number = 3;
    req.entries.resize(2);

    KvBatchRetrieveProtocol proto;
    std::vector<std::uint32_t> target(64, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("must equal entries.size()"), std::string::npos);
}

TEST_F(KvProtocolPackTest, BatchRetrieveValidateRequestRejectsUnalignedLength)
{
    KvBatchRetrieveRequest req;
    req.batch_number = 1;
    req.entries.resize(1);
    req.entries[0].key = MakeCacheKey("key");
    req.entries[0].offset = 0;
    req.entries[0].buffer_addr = 0x1000;
    req.entries[0].length = 100;

    KvBatchRetrieveProtocol proto;
    std::vector<std::uint32_t> target(64, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("512B aligned"), std::string::npos);
}

TEST_F(KvProtocolPackTest, BatchRetrieveValidateRequestRejectsAllZeroKey)
{
    KvBatchRetrieveRequest req;
    req.batch_number = 1;
    req.entries.resize(1);
    req.entries[0].key = CacheKey{};
    req.entries[0].offset = 0;
    req.entries[0].buffer_addr = 0x1000;
    req.entries[0].length = 512;

    KvBatchRetrieveProtocol proto;
    std::vector<std::uint32_t> target(64, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("all zeros"), std::string::npos);
}

TEST_F(KvProtocolPackTest, DeleteValidateRequestAcceptsValid)
{
    KvDeleteRequest req;
    req.batch_number = 2;
    req.keys = {MakeCacheKey("key1"), MakeCacheKey("key2")};

    KvDeleteProtocol proto;
    std::vector<std::uint32_t> target(32, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_TRUE(status.ok()) << status.message;
}

TEST_F(KvProtocolPackTest, DeleteValidateRequestRejectsAllZeroKey)
{
    KvDeleteRequest req;
    req.batch_number = 2;
    req.keys = {MakeCacheKey("key1"), CacheKey{}};

    KvDeleteProtocol proto;
    std::vector<std::uint32_t> target(32, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("all zeros"), std::string::npos);
}

TEST_F(KvProtocolPackTest, ExistValidateRequestAcceptsValid)
{
    KvExistRequest req;
    req.batch_number = 1;
    req.keys = {MakeCacheKey("exist_key")};

    KvExistProtocol proto;
    std::vector<std::uint32_t> target(32, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_TRUE(status.ok()) << status.message;
}

TEST_F(KvProtocolPackTest, ExistValidateRequestRejectsBatchNumberOverflow)
{
    KvExistRequest req;
    req.batch_number = 300;
    req.keys.resize(300, MakeCacheKey("key"));

    KvExistProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("must be in range"), std::string::npos);
}

TEST_F(KvProtocolPackTest, ExistValidateRequestRejectsAllZeroKey)
{
    KvExistRequest req;
    req.batch_number = 1;
    req.keys = {CacheKey{}};

    KvExistProtocol proto;
    std::vector<std::uint32_t> target(32, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("all zeros"), std::string::npos);
}

TEST_F(KvProtocolPackTest, StoreValidateRequestRejectsZeroLength)
{
    KvStoreRequest req;
    req.buffer_addr = 0x1000;
    req.buffer_length = 512;
    req.offset = 0;
    req.length = 0;
    req.key = MakeCacheKey("key");

    KvStoreProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("must be non-zero"), std::string::npos);
}

TEST_F(KvProtocolPackTest, StoreValidateRequestRejectsUnalignedOffset)
{
    KvStoreRequest req;
    req.buffer_addr = 0x1000;
    req.buffer_length = 512;
    req.offset = 100;
    req.length = 1;
    req.key = MakeCacheKey("key");

    KvStoreProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("512B aligned"), std::string::npos);
}

class ProtocolManagerTest : public ::testing::Test {
protected:
    void SetUp() override { mgr_ = std::make_unique<ProtocolManager>(); }

    std::unique_ptr<ProtocolManager> mgr_;
};

TEST_F(ProtocolManagerTest, PackStoreRequest)
{
    KvStoreRequest req;
    req.cid = 0x1234;
    req.kv_ns_id = 1;
    req.dtype = 1;
    req.dspec = 5;
    req.buffer_addr = 0x1000;
    req.buffer_length = 512;
    req.mr_key = 0xABCD;
    req.offset = 0;
    req.length = 1;
    req.key = MakeCacheKey("test_key");

    std::vector<std::uint32_t> target(16, 0);
    auto status = mgr_->PackRequest(target.data(), KvOpcode::Store, req);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(target[0] & 0xFF, 0x01);
    EXPECT_EQ((target[0] >> 16) & 0xFFFF, 0x1234);
    EXPECT_EQ(target[1], 1);
}

TEST_F(ProtocolManagerTest, PackBatchStoreRequest)
{
    KvBatchStoreRequest req;
    req.cid = 0x5678;
    req.kv_ns_id = 2;
    req.dtype = 1;
    req.dspec = 0;
    req.response_buffer_addr = 0x2000;
    req.response_mr_key = 0x1111;
    req.rflag = true;
    req.batch_number = 2;
    req.entries.resize(2);
    req.entries[0].key = MakeCacheKey("key1");
    req.entries[0].offset = 0;
    req.entries[0].buffer_addr = 0x3000;
    req.entries[0].length = 512;
    req.entries[1].key = MakeCacheKey("key2");
    req.entries[1].offset = 512;
    req.entries[1].buffer_addr = 0x4000;
    req.entries[1].length = 512;

    std::vector<std::uint32_t> target(64, 0);
    auto status = mgr_->PackRequest(target.data(), KvOpcode::BatchStore, req);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(target[0] & 0xFF, 0x45);
    EXPECT_EQ((target[0] >> 16) & 0xFFFF, 0x5678);
    EXPECT_EQ(target[10], 2);
}

TEST_F(ProtocolManagerTest, PackKeepAliveRequest)
{
    KvKeepAliveRequest req;
    req.cid = 0xAAAA;
    req.response_buffer_addr = 0x5000;
    req.response_mr_key = 0x2222;
    req.rflag = true;

    std::vector<std::uint32_t> target(16, 0);
    auto status = mgr_->PackRequest(target.data(), KvOpcode::KeepAlive, req);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(target[0] & 0xFF, 0xF4);
    EXPECT_EQ((target[0] >> 16) & 0xFFFF, 0xAAAA);
}

TEST_F(ProtocolManagerTest, PackRequestRejectsInvalidRequest)
{
    KvStoreRequest req;
    req.buffer_addr = 0;
    req.buffer_length = 512;
    req.offset = 0;
    req.length = 1;
    req.key = MakeCacheKey("key");

    std::vector<std::uint32_t> target(16, 0);
    auto status = mgr_->PackRequest(target.data(), KvOpcode::Store, req);
    ASSERT_FALSE(status.ok());
    EXPECT_NE(status.message.find("buffer_addr is zero"), std::string::npos);
}

TEST_F(ProtocolManagerTest, PollResponseCid)
{
    std::uint32_t cqe_data[8] = {0};
    cqe_data[3] = 0x9ABC | (0x123 << 17);

    std::uint16_t cid = 0;
    auto status = mgr_->PollResponseCid(cqe_data, cid);
    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(cid, 0x9ABC);
}

TEST_F(ProtocolManagerTest, UnpackBatchStoreResponse)
{
    std::uint32_t cqe_data[8] = {0};
    cqe_data[3] = 0x1111 | (0x456 << 17);
    cqe_data[4] = 0x0 | (0x1 << 4) | (0x3 << 8);

    KvResponse resp;
    auto status = mgr_->UnpackResponse(cqe_data, KvOpcode::BatchStore, 3, resp);
    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(resp.cid, 0x1111);
    EXPECT_EQ(resp.status, 0x456);
    ASSERT_EQ(resp.result_buffer.size(), 3);
    EXPECT_EQ(resp.result_buffer[0], 0x0);
    EXPECT_EQ(resp.result_buffer[1], 0x1);
    EXPECT_EQ(resp.result_buffer[2], 0x3);
}

TEST_F(ProtocolManagerTest, UnpackDeleteResponse)
{
    std::uint32_t cqe_data[8] = {0};
    cqe_data[3] = 0x2222;
    cqe_data[4] = 0x5;

    KvResponse resp;
    auto status = mgr_->UnpackResponse(cqe_data, KvOpcode::Delete, 3, resp);
    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(resp.cid, 0x2222);
    ASSERT_EQ(resp.result_buffer.size(), 3);
    EXPECT_EQ(resp.result_buffer[0], 1);
    EXPECT_EQ(resp.result_buffer[1], 0);
    EXPECT_EQ(resp.result_buffer[2], 1);
}

TEST_F(ProtocolManagerTest, UnpackExistResponse)
{
    std::uint32_t cqe_data[8] = {0};
    cqe_data[0] = 0x0003;
    cqe_data[3] = 0x3333;
    cqe_data[4] = 0x7;

    KvResponse resp;
    auto status = mgr_->UnpackResponse(cqe_data, KvOpcode::Exist, 3, resp);
    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(resp.cid, 0x3333);
    EXPECT_EQ(resp.existing_key_number, 3);
    ASSERT_EQ(resp.result_buffer.size(), 3);
    EXPECT_EQ(resp.result_buffer[0], 1);
    EXPECT_EQ(resp.result_buffer[1], 1);
    EXPECT_EQ(resp.result_buffer[2], 1);
}

TEST_F(ProtocolManagerTest, GetPackedSizeReturnsCorrectValue)
{
    KvStoreRequest store_req;
    store_req.key = MakeCacheKey("key");
    EXPECT_EQ(mgr_->GetPackedSize(KvOpcode::Store, store_req), 64);

    KvBatchStoreRequest batch_req;
    batch_req.batch_number = 3;
    batch_req.entries.resize(3);
    EXPECT_EQ(mgr_->GetPackedSize(KvOpcode::BatchStore, batch_req), (16 + 3 * 9) * 4);
}

TEST_F(ProtocolManagerTest, EndToEndPackAndUnpack)
{
    KvBatchStoreRequest req;
    req.cid = 0xBEEF;
    req.kv_ns_id = 1;
    req.batch_number = 2;
    req.response_buffer_addr = 0x8000;
    req.response_mr_key = 0x3333;
    req.rflag = true;
    req.entries.resize(2);
    req.entries[0].key = MakeCacheKey("key_a");
    req.entries[0].offset = 0;
    req.entries[0].buffer_addr = 0x9000;
    req.entries[0].length = 512;
    req.entries[1].key = MakeCacheKey("key_b");
    req.entries[1].offset = 512;
    req.entries[1].buffer_addr = 0xA000;
    req.entries[1].length = 512;

    std::vector<std::uint32_t> send_data(64, 0);
    auto pack_status = mgr_->PackRequest(send_data.data(), KvOpcode::BatchStore, req);
    ASSERT_TRUE(pack_status.ok()) << pack_status.message;

    EXPECT_EQ(send_data[0] & 0xFF, 0x45);
    EXPECT_EQ((send_data[0] >> 16) & 0xFFFF, 0xBEEF);

    std::uint32_t flag_data[8] = {0};
    flag_data[3] = 0xBEEF | (0 << 17);
    flag_data[4] = 0x0 | (0x0 << 4);

    std::uint16_t polled_cid = 0;
    auto poll_status = mgr_->PollResponseCid(flag_data, polled_cid);
    ASSERT_TRUE(poll_status.ok());
    EXPECT_EQ(polled_cid, 0xBEEF);

    KvResponse resp;
    auto unpack_status = mgr_->UnpackResponse(flag_data, KvOpcode::BatchStore, 2, resp);
    ASSERT_TRUE(unpack_status.ok()) << unpack_status.message;
    EXPECT_EQ(resp.cid, 0xBEEF);
    EXPECT_EQ(resp.status, 0);
    ASSERT_EQ(resp.result_buffer.size(), 2);
    EXPECT_EQ(resp.result_buffer[0], 0x0);
    EXPECT_EQ(resp.result_buffer[1], 0x0);
}

class KvProtocolVerifyTest : public ::testing::Test {
protected:
    void SetUp() override { mgr_ = std::make_unique<ProtocolManager>(); }
    void TearDown() override {}

    std::unique_ptr<ProtocolManager> mgr_;

    static std::vector<std::uint32_t> PackStore(std::uint16_t cid = 0x1234, std::uint32_t ns = 1,
                                                std::uint8_t dtype = 1, std::uint8_t dspec = 5)
    {
        KvStoreRequest req;
        req.cid = cid;
        req.kv_ns_id = ns;
        req.dtype = dtype;
        req.dspec = dspec;
        req.buffer_addr = 0x123456789ABCULL;
        req.buffer_length = 0x10000;
        req.mr_key = 0x76543210;
        req.offset = 0x1000;
        req.lr = true;
        req.length = 2;
        req.key = MakeCacheKey("test_key_01");
        KvStoreProtocol proto;
        std::vector<std::uint32_t> buf(16, 0);
        auto s = proto.PackSqe(req, buf.data());
        EXPECT_TRUE(s.ok()) << s.message;
        return buf;
    }

    static std::vector<std::uint32_t> PackRetrieve()
    {
        KvRetrieveRequest req;
        req.cid = 0x5678;
        req.kv_ns_id = 2;
        req.buffer_addr = 0xFEDCBA987654ULL;
        req.buffer_length = 0x20000;
        req.mr_key = 0x12345678;
        req.offset = 0x2000;
        req.lr = false;
        req.length = 3;
        req.key = MakeCacheKey("retrieve_key");
        KvRetrieveProtocol proto;
        std::vector<std::uint32_t> buf(16, 0);
        auto s = proto.PackSqe(req, buf.data());
        EXPECT_TRUE(s.ok()) << s.message;
        return buf;
    }

    static std::vector<std::uint32_t> PackBatchStore(std::uint16_t batch_n = 2, bool rflag = false)
    {
        KvBatchStoreRequest req;
        req.cid = 0xABCD;
        req.kv_ns_id = 3;
        req.dtype = 2;
        req.dspec = 10;
        req.response_buffer_addr = rflag ? 0xAAAA0000ULL : 0;
        req.response_mr_key = rflag ? 0x1111 : 0;
        req.lr = false;
        req.rflag = rflag;
        req.batch_number = batch_n;
        for (std::uint16_t i = 0; i < batch_n; ++i) {
            KvBatchStoreEntry e;
            e.offset = (i + 1) * 512;
            e.key = MakeCacheKey(static_cast<std::uint64_t>(0xB500 + i));
            e.buffer_addr = 0xBBBB0000ULL + i * 0x1000;
            e.mr_key = 0x2222 + i;
            e.length = 512;
            req.entries.push_back(e);
        }
        KvBatchStoreProtocol proto;
        std::size_t sz = proto.PackedSize(req) / sizeof(std::uint32_t);
        std::vector<std::uint32_t> buf(sz, 0);
        auto s = proto.PackSqe(req, buf.data());
        EXPECT_TRUE(s.ok()) << s.message;
        return buf;
    }

    static std::vector<std::uint32_t> PackBatchRetrieve(std::uint16_t batch_n = 2,
                                                        bool rflag = false)
    {
        KvBatchRetrieveRequest req;
        req.cid = 0xDCBA;
        req.kv_ns_id = 4;
        req.response_buffer_addr = rflag ? 0xCCCC0000ULL : 0;
        req.response_mr_key = rflag ? 0x3333 : 0;
        req.lr = false;
        req.rflag = rflag;
        req.batch_number = batch_n;
        for (std::uint16_t i = 0; i < batch_n; ++i) {
            KvBatchRetrieveEntry e;
            e.offset = (i + 1) * 512;
            e.key = MakeCacheKey(static_cast<std::uint64_t>(0xB600 + i));
            e.buffer_addr = 0xDDDD0000ULL + i * 0x1000;
            e.mr_key = 0x4444 + i;
            e.length = 512;
            req.entries.push_back(e);
        }
        KvBatchRetrieveProtocol proto;
        std::size_t sz = proto.PackedSize(req) / sizeof(std::uint32_t);
        std::vector<std::uint32_t> buf(sz, 0);
        auto s = proto.PackSqe(req, buf.data());
        EXPECT_TRUE(s.ok()) << s.message;
        return buf;
    }

    static std::vector<std::uint32_t> PackDelete(std::uint16_t batch_n = 3, bool rflag = false)
    {
        KvDeleteRequest req;
        req.cid = 0x1111;
        req.kv_ns_id = 5;
        req.response_buffer_addr = rflag ? 0xEEEE0000ULL : 0;
        req.response_mr_key = rflag ? 0x5555 : 0;
        req.rflag = rflag;
        req.batch_number = batch_n;
        for (std::uint16_t i = 0; i < batch_n; ++i) {
            req.keys.push_back(MakeCacheKey(static_cast<std::uint64_t>(0xD000 + i)));
        }
        KvDeleteProtocol proto;
        std::size_t sz = proto.PackedSize(req) / sizeof(std::uint32_t);
        std::vector<std::uint32_t> buf(sz, 0);
        auto s = proto.PackSqe(req, buf.data());
        EXPECT_TRUE(s.ok()) << s.message;
        return buf;
    }

    static std::vector<std::uint32_t> PackExist(std::uint16_t batch_n = 3, bool rflag = false,
                                                bool sc = false)
    {
        KvExistRequest req;
        req.cid = 0x2222;
        req.kv_ns_id = 6;
        req.response_buffer_addr = rflag ? 0xFFFF0000ULL : 0;
        req.response_mr_key = rflag ? 0x6666 : 0;
        req.rflag = rflag;
        req.sc = sc;
        req.batch_number = batch_n;
        for (std::uint16_t i = 0; i < batch_n; ++i) {
            req.keys.push_back(MakeCacheKey(static_cast<std::uint64_t>(0xE000 + i)));
        }
        KvExistProtocol proto;
        std::size_t sz = proto.PackedSize(req) / sizeof(std::uint32_t);
        std::vector<std::uint32_t> buf(sz, 0);
        auto s = proto.PackSqe(req, buf.data());
        EXPECT_TRUE(s.ok()) << s.message;
        return buf;
    }

    static std::vector<std::uint32_t> PackKeepAlive(bool rflag = false)
    {
        KvKeepAliveRequest req;
        req.cid = 0x3333;
        req.response_buffer_addr = rflag ? 0xAAAA0000ULL : 0;
        req.response_mr_key = rflag ? 0x7777 : 0;
        req.rflag = rflag;
        KvKeepAliveProtocol proto;
        std::vector<std::uint32_t> buf(16, 0);
        auto s = proto.PackSqe(req, buf.data());
        EXPECT_TRUE(s.ok()) << s.message;
        return buf;
    }
};

// ── Valid pack-then-verify for all 7 protocols ──

TEST_F(KvProtocolVerifyTest, StoreValidPackedBufferPasses)
{
    auto buf = PackStore();
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_TRUE(s.ok()) << s.message;
}

TEST_F(KvProtocolVerifyTest, RetrieveValidPackedBufferPasses)
{
    auto buf = PackRetrieve();
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_TRUE(s.ok()) << s.message;
}

TEST_F(KvProtocolVerifyTest, BatchStoreValidPackedBufferPasses)
{
    auto buf = PackBatchStore(2, false);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_TRUE(s.ok()) << s.message;
}

TEST_F(KvProtocolVerifyTest, BatchStoreWithRflagValidPackedBufferPasses)
{
    auto buf = PackBatchStore(3, true);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_TRUE(s.ok()) << s.message;
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveValidPackedBufferPasses)
{
    auto buf = PackBatchRetrieve(2, false);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_TRUE(s.ok()) << s.message;
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveWithRflagValidPackedBufferPasses)
{
    auto buf = PackBatchRetrieve(4, true);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_TRUE(s.ok()) << s.message;
}

TEST_F(KvProtocolVerifyTest, DeleteValidPackedBufferPasses)
{
    auto buf = PackDelete(3, false);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_TRUE(s.ok()) << s.message;
}

TEST_F(KvProtocolVerifyTest, DeleteWithRflagValidPackedBufferPasses)
{
    auto buf = PackDelete(5, true);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_TRUE(s.ok()) << s.message;
}

TEST_F(KvProtocolVerifyTest, ExistValidPackedBufferPasses)
{
    auto buf = PackExist(3, false, false);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_TRUE(s.ok()) << s.message;
}

TEST_F(KvProtocolVerifyTest, ExistWithScAndRflagValidPackedBufferPasses)
{
    auto buf = PackExist(4, true, true);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_TRUE(s.ok()) << s.message;
}

TEST_F(KvProtocolVerifyTest, KeepAliveValidPackedBufferPasses)
{
    auto buf = PackKeepAlive(false);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_TRUE(s.ok()) << s.message;
}

TEST_F(KvProtocolVerifyTest, KeepAliveWithRflagValidPackedBufferPasses)
{
    auto buf = PackKeepAlive(true);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_TRUE(s.ok()) << s.message;
}

// ── ProtocolManager dispatch ──

TEST_F(KvProtocolVerifyTest, ManagerRejectsNullPointer)
{
    auto s = mgr_->VerifyPackedBuffer(nullptr, 64);
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("invalid data_ptr"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, ManagerRejectsTooSmallLength)
{
    std::uint32_t buf[4] = {0};
    auto s = mgr_->VerifyPackedBuffer(buf, 16);
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("length too small"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, ManagerRejectsUnknownOpcode)
{
    std::vector<std::uint32_t> buf(16, 0);
    buf[0] = 0xFF;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("unknown opcode"), std::string::npos);
}

// ── Store: length mismatch ──

TEST_F(KvProtocolVerifyTest, StoreRejectsWrongLength)
{
    auto buf = PackStore();
    auto s = mgr_->VerifyPackedBuffer(buf.data(), (buf.size() - 1) * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("length"), std::string::npos);
}

// ── Store: fixed bits ──

TEST_F(KvProtocolVerifyTest, StoreRejectsBadFixedBits)
{
    auto buf = PackStore();
    buf[0] &= ~(0x3U << 14);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("fixed bits"), std::string::npos);
}

// ── Store: reserved bits in data[0] ──

TEST_F(KvProtocolVerifyTest, StoreRejectsReservedBitsInDword0)
{
    auto buf = PackStore();
    buf[0] |= (1U << 10);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── Store: reserved bits in data[2] ──

TEST_F(KvProtocolVerifyTest, StoreRejectsReservedBitsInDword2)
{
    auto buf = PackStore();
    buf[2] |= 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── Store: reserved bits in data[3-5] ──

TEST_F(KvProtocolVerifyTest, StoreRejectsReservedBitsInDword3)
{
    auto buf = PackStore();
    buf[3] = 0xDEAD;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── Store: buffer_addr zero ──

TEST_F(KvProtocolVerifyTest, StoreRejectsZeroBufferAddr)
{
    auto buf = PackStore();
    buf[6] = 0;
    buf[7] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("buffer_addr"), std::string::npos);
}

// ── Store: buffer_length zero ──

TEST_F(KvProtocolVerifyTest, StoreRejectsZeroBufferLength)
{
    auto buf = PackStore();
    buf[8] &= 0xFF000000U;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("buffer_length"), std::string::npos);
}

// ── Store: buffer_length not aligned ──

TEST_F(KvProtocolVerifyTest, StoreRejectsUnalignedBufferLength)
{
    auto buf = PackStore();
    buf[8] = (buf[8] & 0xFF000000U) | 0x100;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("512B aligned"), std::string::npos);
}

// ── Store: DptrType mismatch ──

TEST_F(KvProtocolVerifyTest, StoreRejectsBadDptrType)
{
    auto buf = PackStore();
    buf[9] = (0x01U << 24) | (buf[9] & 0xFFFFFF);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("DptrType"), std::string::npos);
}

// ── Store: offset not aligned ──

TEST_F(KvProtocolVerifyTest, StoreRejectsUnalignedOffset)
{
    auto buf = PackStore();
    buf[10] = 0x100;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("offset"), std::string::npos);
}

// ── Store: length zero ──

TEST_F(KvProtocolVerifyTest, StoreRejectsZeroLength)
{
    auto buf = PackStore();
    buf[11] &= 0xFF000000U;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("length"), std::string::npos);
}

// ── Store: reserved bits in data[11] ──

TEST_F(KvProtocolVerifyTest, StoreRejectsReservedBitsInDword11)
{
    auto buf = PackStore();
    buf[11] |= (1U << 25);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── Store: key all zeros ──

TEST_F(KvProtocolVerifyTest, StoreRejectsAllZeroKey)
{
    auto buf = PackStore();
    buf[12] = 0;
    buf[13] = 0;
    buf[14] = 0;
    buf[15] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("key"), std::string::npos);
}

// ── Retrieve: length mismatch ──

TEST_F(KvProtocolVerifyTest, RetrieveRejectsWrongLength)
{
    auto buf = PackRetrieve();
    auto s = mgr_->VerifyPackedBuffer(buf.data(), (buf.size() + 1) * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("length"), std::string::npos);
}

// ── Retrieve: reserved bits in data[2] ──

TEST_F(KvProtocolVerifyTest, RetrieveRejectsReservedBitsInDword2)
{
    auto buf = PackRetrieve();
    buf[2] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── Retrieve: DptrType mismatch ──

TEST_F(KvProtocolVerifyTest, RetrieveRejectsBadDptrType)
{
    auto buf = PackRetrieve();
    buf[9] = (0x01U << 24) | (buf[9] & 0xFFFFFF);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("DptrType"), std::string::npos);
}

// ── BatchStore: length mismatch ──

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsWrongLength)
{
    auto buf = PackBatchStore(2);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), (buf.size() + 4) * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("length"), std::string::npos);
}

// ── BatchStore: batch_number out of range ──

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsZeroBatchNumber)
{
    auto buf = PackBatchStore(2);
    buf[10] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("batch_number"), std::string::npos);
}

// ── BatchStore: data[8] mismatch ──

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsData8Mismatch)
{
    auto buf = PackBatchStore(2);
    buf[8] = 0xDEAD;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("data[8]"), std::string::npos);
}

// ── BatchStore: DptrType mismatch ──

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsBadDptrType)
{
    auto buf = PackBatchStore(2);
    buf[9] = (0x40U << 24);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("DptrType"), std::string::npos);
}

// ── BatchStore: reserved bits in data[10] ──

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsReservedBitsInDword10)
{
    auto buf = PackBatchStore(2);
    buf[10] |= (1U << 20);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── BatchStore: reserved bits in data[12-15] ──

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsReservedBitsInDword12)
{
    auto buf = PackBatchStore(2);
    buf[12] = 0xBEEF;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── BatchStore: rflag false but response addr non-zero ──

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsRflagFalseWithResponseAddr)
{
    auto buf = PackBatchStore(2, false);
    buf[3] = 0x1234;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

// ── BatchStore: entry key all zeros ──

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsEntryAllZeroKey)
{
    auto buf = PackBatchStore(2);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[1] = 0;
    entry0[2] = 0;
    entry0[3] = 0;
    entry0[4] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("key"), std::string::npos);
}

// ── BatchStore: entry buffer_addr zero ──

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsEntryZeroBufferAddr)
{
    auto buf = PackBatchStore(2);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[5] = 0;
    entry0[6] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("buffer_addr"), std::string::npos);
}

// ── BatchStore: entry length zero ──

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsEntryZeroLength)
{
    auto buf = PackBatchStore(2);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[7] &= 0xFF000000U;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("length"), std::string::npos);
}

// ── BatchStore: entry DptrType mismatch ──

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsEntryBadDptrType)
{
    auto buf = PackBatchStore(2);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[8] = (0x01U << 24) | (entry0[8] & 0xFFFFFF);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("DptrType"), std::string::npos);
}

// ── BatchRetrieve: reserved bits in data[2] ──

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsReservedBitsInDword2)
{
    auto buf = PackBatchRetrieve(2);
    buf[2] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── BatchRetrieve: data[8] mismatch ──

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsData8Mismatch)
{
    auto buf = PackBatchRetrieve(2);
    buf[8] = 0xBEEF;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("data[8]"), std::string::npos);
}

// ── BatchRetrieve: entry offset not aligned ──

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsEntryUnalignedOffset)
{
    auto buf = PackBatchRetrieve(2);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[0] = 0x100;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("offset"), std::string::npos);
}

// ── Delete: data[8] mismatch ──

TEST_F(KvProtocolVerifyTest, DeleteRejectsData8Mismatch)
{
    auto buf = PackDelete(3);
    buf[8] = 0xDEAD;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("data[8]"), std::string::npos);
}

// ── Delete: batch_number out of range ──

TEST_F(KvProtocolVerifyTest, DeleteRejectsBatchNumberOutOfRange)
{
    auto buf = PackDelete(3);
    buf[10] = 255;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("batch_number"), std::string::npos);
}

// ── Delete: reserved bits in data[11-15] ──

TEST_F(KvProtocolVerifyTest, DeleteRejectsReservedBitsInDword11)
{
    auto buf = PackDelete(3);
    buf[11] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, DeleteRejectsReservedBitsInDword15)
{
    auto buf = PackDelete(3);
    buf[15] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── Delete: entry key all zeros ──

TEST_F(KvProtocolVerifyTest, DeleteRejectsEntryAllZeroKey)
{
    auto buf = PackDelete(3);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[0] = 0;
    entry0[1] = 0;
    entry0[2] = 0;
    entry0[3] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("key"), std::string::npos);
}

// ── Exist: data[8] mismatch ──

TEST_F(KvProtocolVerifyTest, ExistRejectsData8Mismatch)
{
    auto buf = PackExist(3);
    buf[8] = 0xDEAD;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("data[8]"), std::string::npos);
}

// ── Exist: batch_number out of range ──

TEST_F(KvProtocolVerifyTest, ExistRejectsBatchNumberOutOfRange)
{
    auto buf = PackExist(3);
    buf[10] = 257;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("batch_number"), std::string::npos);
}

// ── Exist: reserved bits in data[11] ──

TEST_F(KvProtocolVerifyTest, ExistRejectsReservedBitsInDword11)
{
    auto buf = PackExist(3);
    buf[11] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── Exist: reserved bits in data[12-15] ──

TEST_F(KvProtocolVerifyTest, ExistRejectsReservedBitsInDword12)
{
    auto buf = PackExist(3);
    buf[12] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── Exist: reserved bits in data[10] bit17-31 ──

TEST_F(KvProtocolVerifyTest, ExistRejectsReservedBitsInDword10High)
{
    auto buf = PackExist(3, false, true);
    buf[10] |= (1U << 20);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── KeepAlive: length mismatch ──

TEST_F(KvProtocolVerifyTest, KeepAliveRejectsWrongLength)
{
    auto buf = PackKeepAlive();
    auto s = mgr_->VerifyPackedBuffer(buf.data(), (buf.size() + 1) * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("length"), std::string::npos);
}

// ── KeepAlive: reserved bits in data[0] ──

TEST_F(KvProtocolVerifyTest, KeepAliveRejectsReservedBitsInDword0)
{
    auto buf = PackKeepAlive();
    buf[0] |= (1U << 10);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── KeepAlive: reserved bits in data[14-15] ──

TEST_F(KvProtocolVerifyTest, KeepAliveRejectsReservedBitsBit14)
{
    auto buf = PackKeepAlive();
    buf[0] |= (1U << 14);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── KeepAlive: reserved bits in data[1-2] ──

TEST_F(KvProtocolVerifyTest, KeepAliveRejectsReservedBitsInDword1)
{
    auto buf = PackKeepAlive();
    buf[1] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── KeepAlive: reserved bits in data[6-15] ──

TEST_F(KvProtocolVerifyTest, KeepAliveRejectsReservedBitsInDword6)
{
    auto buf = PackKeepAlive();
    buf[6] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, KeepAliveRejectsReservedBitsInDword15)
{
    auto buf = PackKeepAlive();
    buf[15] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

// ── KeepAlive: rflag false but response addr non-zero ──

TEST_F(KvProtocolVerifyTest, KeepAliveRejectsRflagFalseWithResponseAddr)
{
    auto buf = PackKeepAlive(false);
    buf[3] = 0x1234;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

// ══════════════════════════════════════════════════════════════
// Supplementary tests for full branch coverage (store.xlsx spec)
// ══════════════════════════════════════════════════════════════

// ── Retrieve: missing branches ──

TEST_F(KvProtocolVerifyTest, RetrieveRejectsBadFixedBits)
{
    auto buf = PackRetrieve();
    buf[0] &= ~(0x3U << 14);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("fixed bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, RetrieveRejectsReservedBitsInDword0)
{
    auto buf = PackRetrieve();
    buf[0] |= (1U << 10);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, RetrieveRejectsReservedBitsInDword3)
{
    auto buf = PackRetrieve();
    buf[3] = 0xDEAD;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, RetrieveRejectsZeroBufferAddr)
{
    auto buf = PackRetrieve();
    buf[6] = 0;
    buf[7] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("buffer_addr"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, RetrieveRejectsZeroBufferLength)
{
    auto buf = PackRetrieve();
    buf[8] &= 0xFF000000U;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("buffer_length"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, RetrieveRejectsUnalignedBufferLength)
{
    auto buf = PackRetrieve();
    buf[8] = (buf[8] & 0xFF000000U) | 0x100;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("512B aligned"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, RetrieveRejectsUnalignedOffset)
{
    auto buf = PackRetrieve();
    buf[10] = 0x100;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("offset"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, RetrieveRejectsZeroLength)
{
    auto buf = PackRetrieve();
    buf[11] &= 0xFF000000U;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("length"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, RetrieveRejectsReservedBitsInDword11)
{
    auto buf = PackRetrieve();
    buf[11] |= (1U << 25);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, RetrieveRejectsAllZeroKey)
{
    auto buf = PackRetrieve();
    buf[12] = 0;
    buf[13] = 0;
    buf[14] = 0;
    buf[15] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("key"), std::string::npos);
}

// ── BatchStore: missing branches ──

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsBadFixedBits)
{
    auto buf = PackBatchStore(2);
    buf[0] &= ~(0x3U << 14);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("fixed bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsReservedBitsInDword0)
{
    auto buf = PackBatchStore(2);
    buf[0] |= (1U << 10);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsReservedBitsInDword2)
{
    auto buf = PackBatchStore(2);
    buf[2] |= 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsReservedBitsInDword6)
{
    auto buf = PackBatchStore(2);
    buf[6] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsRflagSetWithZeroResponseAddr)
{
    auto buf = PackBatchStore(2, true);
    buf[3] = 0;
    buf[4] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsRflagSetWithZeroResponseMrKey)
{
    auto buf = PackBatchStore(2, true);
    buf[5] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsRflagFalseWithResponseMrKey)
{
    auto buf = PackBatchStore(2, false);
    buf[5] = 0x1234;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsEntryUnalignedOffset)
{
    auto buf = PackBatchStore(2);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[0] = 0x100;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("offset"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchStoreRejectsEntryUnalignedLength)
{
    auto buf = PackBatchStore(2);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[7] = (entry0[7] & 0xFF000000U) | 0x100;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("512B aligned"), std::string::npos);
}

// ── BatchRetrieve: missing branches ──

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsBadFixedBits)
{
    auto buf = PackBatchRetrieve(2);
    buf[0] &= ~(0x3U << 14);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("fixed bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsReservedBitsInDword0)
{
    auto buf = PackBatchRetrieve(2);
    buf[0] |= (1U << 10);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsReservedBitsInDword6)
{
    auto buf = PackBatchRetrieve(2);
    buf[6] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsRflagSetWithZeroResponseAddr)
{
    auto buf = PackBatchRetrieve(2, true);
    buf[3] = 0;
    buf[4] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsRflagSetWithZeroResponseMrKey)
{
    auto buf = PackBatchRetrieve(2, true);
    buf[5] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsRflagFalseWithResponseAddr)
{
    auto buf = PackBatchRetrieve(2, false);
    buf[3] = 0x1234;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsRflagFalseWithResponseMrKey)
{
    auto buf = PackBatchRetrieve(2, false);
    buf[5] = 0x1234;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsBadDptrType)
{
    auto buf = PackBatchRetrieve(2);
    buf[9] = (0x40U << 24);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("DptrType"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsReservedBitsInDword10)
{
    auto buf = PackBatchRetrieve(2);
    buf[10] |= (1U << 20);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsZeroBatchNumber)
{
    auto buf = PackBatchRetrieve(2);
    buf[10] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("batch_number"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsWrongLength)
{
    auto buf = PackBatchRetrieve(2);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), (buf.size() + 4) * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("length"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsReservedBitsInDword11)
{
    auto buf = PackBatchRetrieve(2);
    buf[11] |= (1U << 10);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsEntryAllZeroKey)
{
    auto buf = PackBatchRetrieve(2);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[1] = 0;
    entry0[2] = 0;
    entry0[3] = 0;
    entry0[4] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("key"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsEntryZeroBufferAddr)
{
    auto buf = PackBatchRetrieve(2);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[5] = 0;
    entry0[6] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("buffer_addr"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsEntryZeroLength)
{
    auto buf = PackBatchRetrieve(2);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[7] &= 0xFF000000U;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("length"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsEntryUnalignedLength)
{
    auto buf = PackBatchRetrieve(2);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[7] = (entry0[7] & 0xFF000000U) | 0x100;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("512B aligned"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, BatchRetrieveRejectsEntryBadDptrType)
{
    auto buf = PackBatchRetrieve(2);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[8] = (0x01U << 24) | (entry0[8] & 0xFFFFFF);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("DptrType"), std::string::npos);
}

// ── Delete: missing branches ──

TEST_F(KvProtocolVerifyTest, DeleteRejectsBadFixedBits)
{
    auto buf = PackDelete(3);
    buf[0] &= ~(0x3U << 14);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("fixed bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, DeleteRejectsReservedBitsInDword0)
{
    auto buf = PackDelete(3);
    buf[0] |= (1U << 10);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, DeleteRejectsReservedBitsInDword2)
{
    auto buf = PackDelete(3);
    buf[2] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, DeleteRejectsReservedBitsInDword6)
{
    auto buf = PackDelete(3);
    buf[6] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, DeleteRejectsRflagSetWithZeroResponseAddr)
{
    auto buf = PackDelete(3, true);
    buf[3] = 0;
    buf[4] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, DeleteRejectsRflagSetWithZeroResponseMrKey)
{
    auto buf = PackDelete(3, true);
    buf[5] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, DeleteRejectsRflagFalseWithResponseAddr)
{
    auto buf = PackDelete(3, false);
    buf[3] = 0x1234;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, DeleteRejectsRflagFalseWithResponseMrKey)
{
    auto buf = PackDelete(3, false);
    buf[5] = 0x1234;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, DeleteRejectsBadDptrType)
{
    auto buf = PackDelete(3);
    buf[9] = (0x40U << 24);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("DptrType"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, DeleteRejectsReservedBitsInDword10)
{
    auto buf = PackDelete(3);
    buf[10] |= (1U << 20);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, DeleteRejectsWrongLength)
{
    auto buf = PackDelete(3);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), (buf.size() + 4) * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("length"), std::string::npos);
}

// ── Exist: missing branches ──

TEST_F(KvProtocolVerifyTest, ExistRejectsBadFixedBits)
{
    auto buf = PackExist(3);
    buf[0] &= ~(0x3U << 14);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("fixed bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, ExistRejectsReservedBitsInDword0)
{
    auto buf = PackExist(3);
    buf[0] |= (1U << 10);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, ExistRejectsReservedBitsInDword2)
{
    auto buf = PackExist(3);
    buf[2] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, ExistRejectsReservedBitsInDword6)
{
    auto buf = PackExist(3);
    buf[6] = 0x1;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reserved bits"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, ExistRejectsRflagSetWithZeroResponseAddr)
{
    auto buf = PackExist(3, true);
    buf[3] = 0;
    buf[4] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, ExistRejectsRflagSetWithZeroResponseMrKey)
{
    auto buf = PackExist(3, true);
    buf[5] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, ExistRejectsRflagFalseWithResponseAddr)
{
    auto buf = PackExist(3, false);
    buf[3] = 0x1234;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, ExistRejectsRflagFalseWithResponseMrKey)
{
    auto buf = PackExist(3, false);
    buf[5] = 0x1234;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, ExistRejectsBadDptrType)
{
    auto buf = PackExist(3);
    buf[9] = (0x40U << 24);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("DptrType"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, ExistRejectsWrongLength)
{
    auto buf = PackExist(3);
    auto s = mgr_->VerifyPackedBuffer(buf.data(), (buf.size() + 4) * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("length"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, ExistRejectsEntryAllZeroKey)
{
    auto buf = PackExist(3);
    std::uint32_t* entry0 = buf.data() + 16;
    entry0[0] = 0;
    entry0[1] = 0;
    entry0[2] = 0;
    entry0[3] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("key"), std::string::npos);
}

// ── KeepAlive: missing branches ──

TEST_F(KvProtocolVerifyTest, KeepAliveRejectsRflagSetWithZeroResponseAddr)
{
    auto buf = PackKeepAlive(true);
    buf[3] = 0;
    buf[4] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, KeepAliveRejectsRflagSetWithZeroResponseMrKey)
{
    auto buf = PackKeepAlive(true);
    buf[5] = 0;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

TEST_F(KvProtocolVerifyTest, KeepAliveRejectsRflagFalseWithResponseMrKey)
{
    auto buf = PackKeepAlive(false);
    buf[5] = 0x1234;
    auto s = mgr_->VerifyPackedBuffer(buf.data(), buf.size() * sizeof(std::uint32_t));
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("rflag"), std::string::npos);
}

}  // namespace
}  // namespace UC::ASU
