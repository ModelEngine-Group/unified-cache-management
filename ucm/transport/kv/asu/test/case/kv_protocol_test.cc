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
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>

namespace UC::ASU {
namespace {

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
    req.key = kKey;

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
    std::size_t key_len = std::min(kKey.size(), static_cast<std::size_t>(16));
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
    req.key = kKey;

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
    std::memcpy(&expected[12], kKey.data(), std::min(kKey.size(), static_cast<std::size_t>(16)));

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
    std::memcpy(&expected[17], entry1.key.data(),
                std::min(entry1.key.size(), static_cast<std::size_t>(16)));
    expected[21] = entry1.buffer_addr & 0xFFFFFFFFULL;
    expected[22] = (entry1.buffer_addr >> 32) & 0xFFFFFFFFULL;
    expected[23] = ((entry1.mr_key & 0xFF) << 24) | (entry1.length & 0xFFFFFF);
    expected[24] = (0x40 << 24) | ((entry1.mr_key >> 8) & 0xFFFFFF);

    expected[25] = entry2.offset;
    std::memcpy(&expected[26], entry2.key.data(),
                std::min(entry2.key.size(), static_cast<std::size_t>(16)));
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
    entry.key = "batch_ret_key";
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
    std::memcpy(&expected[17], entry.key.data(),
                std::min(entry.key.size(), static_cast<std::size_t>(16)));
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
    req.keys = {"delete_key_1", "delete_key_2"};

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
                std::min(req.keys[0].size(), static_cast<std::size_t>(16)));
    std::memcpy(&expected[20], req.keys[1].data(),
                std::min(req.keys[1].size(), static_cast<std::size_t>(16)));

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
    req.keys = {"exist_key"};

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
                std::min(req.keys[0].size(), static_cast<std::size_t>(16)));

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

TEST_F(KvProtocolPackTest, StoreAndRetrieveUnpackCqeReturnsUnsupported)
{
    KvStoreProtocol store_proto;
    KvResponse resp;
    std::uint32_t cqe_data[4] = {0, 0, 0, 0};
    auto status = store_proto.UnpackCqe(cqe_data, 0, resp);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code, StatusCode::UNSUPPORTED);

    KvRetrieveProtocol retrieve_proto;
    status = retrieve_proto.UnpackCqe(cqe_data, 0, resp);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code, StatusCode::UNSUPPORTED);
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
    req.key = "key";

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
    req.key = "key";

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
    req.key = "key";

    KvStoreProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("512B aligned"), std::string::npos);
}

TEST_F(KvProtocolPackTest, StoreValidateRequestRejectsEmptyKey)
{
    KvStoreRequest req;
    req.buffer_addr = 0x1000;
    req.buffer_length = 512;
    req.offset = 0;
    req.length = 1;
    req.key = "";

    KvStoreProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("key is empty"), std::string::npos);
}

TEST_F(KvProtocolPackTest, StoreValidateRequestRejectsKeyTooLong)
{
    KvStoreRequest req;
    req.buffer_addr = 0x1000;
    req.buffer_length = 512;
    req.offset = 0;
    req.length = 1;
    req.key = "this_key_is_way_too_long_for_16_bytes";

    KvStoreProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("key size("), std::string::npos);
    EXPECT_NE(status.message.find("exceeds 16 bytes"), std::string::npos);
}

TEST_F(KvProtocolPackTest, StoreValidateRequestRejectsDtypeOverflow)
{
    KvStoreRequest req;
    req.buffer_addr = 0x1000;
    req.buffer_length = 512;
    req.offset = 0;
    req.length = 1;
    req.key = "key";
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
    req.key = "key";
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
    req.key = "valid_key";
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
    req.entries[0].key = "key";

    KvBatchStoreProtocol proto;
    std::vector<std::uint32_t> target(64, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("response_buffer_addr is zero"), std::string::npos);
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
    req.key = "retrieve_key";

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
    req.key = "key";

    KvRetrieveProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("buffer_addr is zero"), std::string::npos);
}

TEST_F(KvProtocolPackTest, BatchRetrieveValidateRequestAcceptsValid)
{
    KvBatchRetrieveRequest req;
    req.batch_number = 2;
    req.entries.resize(2);
    req.entries[0].key = "key1";
    req.entries[0].offset = 0;
    req.entries[0].buffer_addr = 0x1000;
    req.entries[0].length = 512;
    req.entries[1].key = "key2";
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

TEST_F(KvProtocolPackTest, DeleteValidateRequestAcceptsValid)
{
    KvDeleteRequest req;
    req.batch_number = 2;
    req.keys = {"key1", "key2"};

    KvDeleteProtocol proto;
    std::vector<std::uint32_t> target(32, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_TRUE(status.ok()) << status.message;
}

TEST_F(KvProtocolPackTest, DeleteValidateRequestRejectsEmptyKey)
{
    KvDeleteRequest req;
    req.batch_number = 2;
    req.keys = {"key1", ""};

    KvDeleteProtocol proto;
    std::vector<std::uint32_t> target(32, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("is empty"), std::string::npos);
}

TEST_F(KvProtocolPackTest, ExistValidateRequestAcceptsValid)
{
    KvExistRequest req;
    req.batch_number = 1;
    req.keys = {"exist_key"};

    KvExistProtocol proto;
    std::vector<std::uint32_t> target(32, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_TRUE(status.ok()) << status.message;
}

TEST_F(KvProtocolPackTest, ExistValidateRequestRejectsBatchNumberOverflow)
{
    KvExistRequest req;
    req.batch_number = 300;
    req.keys.resize(300, "key");

    KvExistProtocol proto;
    std::vector<std::uint32_t> target(16, 0);
    auto status = proto.PackSqe(req, target.data());
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message.find("must be in range"), std::string::npos);
}

TEST_F(KvProtocolPackTest, StoreValidateRequestRejectsZeroLength)
{
    KvStoreRequest req;
    req.buffer_addr = 0x1000;
    req.buffer_length = 512;
    req.offset = 0;
    req.length = 0;
    req.key = "key";

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
    req.key = "key";

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
    req.key = "test_key";

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
    req.entries[0].key = "key1";
    req.entries[0].offset = 0;
    req.entries[0].buffer_addr = 0x3000;
    req.entries[0].length = 512;
    req.entries[1].key = "key2";
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
    req.key = "key";

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

TEST_F(ProtocolManagerTest, UnpackStoreReturnsUnsupported)
{
    std::uint32_t cqe_data[4] = {0};

    KvResponse resp;
    auto status = mgr_->UnpackResponse(cqe_data, KvOpcode::Store, 0, resp);
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code, StatusCode::UNSUPPORTED);
}

TEST_F(ProtocolManagerTest, GetPackedSizeReturnsCorrectValue)
{
    KvStoreRequest store_req;
    store_req.key = "key";
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
    req.entries[0].key = "key_a";
    req.entries[0].offset = 0;
    req.entries[0].buffer_addr = 0x9000;
    req.entries[0].length = 512;
    req.entries[1].key = "key_b";
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

}  // namespace
}  // namespace UC::ASU
