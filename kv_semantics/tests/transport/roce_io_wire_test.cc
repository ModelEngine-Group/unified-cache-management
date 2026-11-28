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

// Tests of the CPU data path wire format: how a packed SQE is described and
// rewritten for the send/response mode, and how the store's completion record is
// published into the caller's flag buffer.  Everything here is plain C++, so the
// cases also run on a development host without rdma-core.

#include "roce/roce_io_wire.h"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "kv_types.h"

namespace kv::test {
namespace {

std::uint32_t ResponseStatusDword(std::uint16_t cid, std::uint16_t status)
{
    return (static_cast<std::uint32_t>(cid) & roce::kSqeCidMask) |
           ((static_cast<std::uint32_t>(status) & roce::kCqeStatusMask) << roce::kCqeStatusShift);
}

void StoreDword(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value)
{
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

std::uint32_t LoadDword(const std::vector<std::uint8_t>& bytes, std::size_t offset)
{
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

std::vector<std::uint32_t> MakeBatchSqe(std::uint8_t opcode, std::uint16_t cid,
                                        std::uint16_t batchNumber, bool rflag = true)
{
    std::size_t entryDwords = roce::kSqeBatchEntryDwords;
    if (opcode == roce::kOpcodeDelete || opcode == roce::kOpcodeExist) {
        entryDwords = roce::kSqeKeyEntryDwords;
    }
    std::vector<std::uint32_t> sqe(roce::kSqeHeaderDwords + batchNumber * entryDwords, 0U);
    sqe[0] = (static_cast<std::uint32_t>(cid) << roce::kSqeCidShift) |
             (roce::kSqeFixedBits << 14U) | static_cast<std::uint32_t>(opcode);
    if (rflag) { sqe[0] |= (1U << roce::kSqeRflagBit); }
    // A response buffer as the client packs it for the writeback path.
    sqe[3] = 0x11223344U;
    sqe[4] = 0x55667788U;
    sqe[5] = 0x99AABBCCU;
    sqe[10] = batchNumber;
    return sqe;
}

TEST(CpuKvOverRoceIoWireTest, ResponseBytesFollowTheFlagBufferLayout)
{
    // A batch store reports four bits per entry, so 110 entries need 55 bytes of
    // result behind the sixteen byte header.
    EXPECT_EQ(71U, roce::ResponseBytesFor(roce::kOpcodeBatchStore, 110));
    EXPECT_EQ(71U, roce::ResponseBytesFor(roce::kOpcodeBatchRetrieve, 110));
    // Delete and exist report one bit per key.
    EXPECT_EQ(48U, roce::ResponseBytesFor(roce::kOpcodeDelete, 254));
    EXPECT_EQ(48U, roce::ResponseBytesFor(roce::kOpcodeExist, 256));
    // The single key opcodes carry no per entry result.
    EXPECT_EQ(17U, roce::ResponseBytesFor(roce::kOpcodeStore, 1));
    EXPECT_EQ(17U, roce::ResponseBytesFor(roce::kOpcodeRetrieve, 1));
    EXPECT_EQ(17U, roce::ResponseBytesFor(roce::kOpcodeKeepAlive, 1));
    EXPECT_EQ(0U, roce::ResponseBytesFor(0x99U, 1));
}

TEST(CpuKvOverRoceIoWireTest, DescribeRequestReadsCidOpcodeAndBatch)
{
    const std::vector<std::uint32_t> sqe = MakeBatchSqe(roce::kOpcodeBatchStore, 0x1234U, 3U);
    roce::RequestDescription description;

    const Status status =
        roce::DescribeRequest(sqe.data(), sqe.size() * sizeof(std::uint32_t), description);
    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(0x1234U, description.cid);
    EXPECT_EQ(roce::kOpcodeBatchStore, description.opcode);
    EXPECT_EQ(3U, description.batchNumber);
    EXPECT_EQ(roce::ResponseBytesFor(roce::kOpcodeBatchStore, 3U), description.responseBytes);
    EXPECT_EQ((roce::kSqeHeaderDwords + 3U * roce::kSqeBatchEntryDwords) * sizeof(std::uint32_t),
              description.sqeBytes);
    // The fixture mirrors the writeback packing, which the CPU path rejects.
    EXPECT_TRUE(description.writeback);
}

TEST(CpuKvOverRoceIoWireTest, DescribeRequestAcceptsTheKeepAliveFormat)
{
    // The keep alive request reserves bits 14-15 instead of carrying the fixed
    // bits of the data opcodes.
    std::vector<std::uint32_t> sqe(roce::kSqeHeaderDwords, 0U);
    sqe[0] = (0x33U << roce::kSqeCidShift) | roce::kOpcodeKeepAlive;
    sqe[3] = 0x11111111U;
    sqe[4] = 0x22222222U;
    sqe[5] = 0x33333333U;

    roce::RequestDescription description;
    ASSERT_TRUE(
        roce::DescribeRequest(sqe.data(), sqe.size() * sizeof(std::uint32_t), description).ok());
    EXPECT_EQ(0x33U, description.cid);
    EXPECT_EQ(roce::kOpcodeKeepAlive, description.opcode);
    EXPECT_EQ(roce::kResponseHeaderBytes + 1U, description.responseBytes);
    // Send/response requests carry no writeback flag; the provider relies on it.
    EXPECT_FALSE(description.writeback);
}

TEST(CpuKvOverRoceIoWireTest, DescribeRequestRejectsMalformedRequests)
{
    roce::RequestDescription description;
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, roce::DescribeRequest(nullptr, 64U, description).code);

    const std::vector<std::uint32_t> sqe = MakeBatchSqe(roce::kOpcodeBatchStore, 7U, 2U);

    // A buffer shorter than the SQE header cannot be a request.
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
              roce::DescribeRequest(sqe.data(), 8U, description).code);

    // The format requires the fixed bits above the opcode.
    std::vector<std::uint32_t> wrongBits = sqe;
    wrongBits[0] &= ~(roce::kSqeFixedBits << 14U);
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
              roce::DescribeRequest(wrongBits.data(), wrongBits.size() * sizeof(std::uint32_t),
                                    description)
                  .code);

    // A zero cid can never be matched with an answer.
    std::vector<std::uint32_t> zeroCid = sqe;
    zeroCid[0] &= ~(roce::kSqeCidMask << roce::kSqeCidShift);
    EXPECT_EQ(
        StatusCode::INVALID_ARGUMENT,
        roce::DescribeRequest(zeroCid.data(), zeroCid.size() * sizeof(std::uint32_t), description)
            .code);

    // The opcode decides how the entries are laid out, so an unknown one is not
    // sent on the wire.
    std::vector<std::uint32_t> unknownOpcode(roce::kSqeHeaderDwords, 1U);
    unknownOpcode[0] = (7U << roce::kSqeCidShift) | (roce::kSqeFixedBits << 14U) | 0x99U;
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
              roce::DescribeRequest(unknownOpcode.data(),
                                    unknownOpcode.size() * sizeof(std::uint32_t), description)
                  .code);

    // A batch number of zero would describe no entries at all.
    std::vector<std::uint32_t> zeroBatch = sqe;
    zeroBatch[10] = 0U;
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
              roce::DescribeRequest(zeroBatch.data(), zeroBatch.size() * sizeof(std::uint32_t),
                                    description)
                  .code);

    // Above the opcode limit the buffer size cannot be derived any more.
    const std::vector<std::uint32_t> tooManyKeys =
        MakeBatchSqe(roce::kOpcodeDelete, 7U, roce::kMaxDeleteKeys + 1U);
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
              roce::DescribeRequest(tooManyKeys.data(), tooManyKeys.size() * sizeof(std::uint32_t),
                                    description)
                  .code);

    // A buffer that does not match the announced batch size is not forwarded.
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
              roce::DescribeRequest(sqe.data(), 64U, description).code);
}

TEST(CpuKvOverRoceIoWireTest, PublishResponseCopiesTheCompletionIntoTheFlagBuffer)
{
    std::vector<std::uint8_t> response(roce::kResponseMaxBytes, 0U);
    StoreDword(response, 0U, 5U);
    StoreDword(response, 4U, 6U);
    StoreDword(response, 8U, 7U);
    StoreDword(response, 12U, ResponseStatusDword(0x2345U, 0U));
    // A four bit result of the first entries: the first nibble is a hit.
    response[roce::kResponseHeaderBytes] = 0x0FU;

    std::vector<std::uint8_t> flag(128U, 0xAAU);
    const std::size_t capacity = roce::ResponseBytesFor(roce::kOpcodeBatchStore, 4U);
    ASSERT_GT(flag.size(), capacity);

    const Status status =
        roce::PublishResponse(response.data(), response.size(), flag.data(), capacity);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(5U, LoadDword(flag, 0U));
    EXPECT_EQ(6U, LoadDword(flag, 4U));
    EXPECT_EQ(7U, LoadDword(flag, 8U));
    // The cid and status dword is what the client polls for.
    EXPECT_EQ(ResponseStatusDword(0x2345U, 0U), LoadDword(flag, 12U));
    EXPECT_EQ(0x0FU, flag[roce::kResponseHeaderBytes]);
    // The slot capacity is respected: nothing past it is touched.
    for (std::size_t index = capacity; index < flag.size(); ++index) {
        ASSERT_EQ(0xAAU, flag[index]) << "byte " << index << " was written past the slot";
    }
}

TEST(CpuKvOverRoceIoWireTest, PublishResponseRejectsAShortCompletion)
{
    std::vector<std::uint8_t> tooShort(roce::kResponseHeaderBytes - 1U, 0U);
    std::vector<std::uint8_t> flag(64U, 0U);
    EXPECT_EQ(
        StatusCode::INVALID_ARGUMENT,
        roce::PublishResponse(tooShort.data(), tooShort.size(), flag.data(), flag.size()).code);

    std::vector<std::uint8_t> response(roce::kResponseMaxBytes, 0U);
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
              roce::PublishResponse(response.data(), response.size(), flag.data(),
                                    roce::kResponseHeaderBytes - 1U)
                  .code);
}

TEST(CpuKvOverRoceIoWireTest, PublishResponseStaysInsideASmallerSlot)
{
    std::vector<std::uint8_t> response(roce::kResponseMaxBytes, 0U);
    StoreDword(response, 12U, ResponseStatusDword(9U, 0U));
    std::vector<std::uint8_t> payload(roce::kResponseResultBytes, 0x5AU);
    std::memcpy(response.data() + roce::kResponseHeaderBytes, payload.data(), payload.size());

    std::vector<std::uint8_t> flag(32U, 0U);
    const std::size_t capacity = roce::kResponseHeaderBytes + 4U;
    ASSERT_TRUE(
        roce::PublishResponse(response.data(), response.size(), flag.data(), capacity).ok());
    EXPECT_EQ(0x5AU, flag[roce::kResponseHeaderBytes]);
    EXPECT_EQ(0x5AU, flag[roce::kResponseHeaderBytes + 3U]);
    // The next four bytes belong to whatever lives behind the slot.
    EXPECT_EQ(0U, flag[capacity]);
    EXPECT_EQ(0U, flag[flag.size() - 1U]);
}

}  // namespace
}  // namespace kv::test
