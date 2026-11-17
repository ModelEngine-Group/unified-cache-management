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
#include "cache/token_codec.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace UC::Store {
namespace {

TEST(TokenCodecTest, TokenPayloadIsPureSlotTranslation)
{
    // Three slots encode to exactly twelve bytes, four per slot.
    const std::vector<std::uint32_t> slots = {0, 3, 7};
    const std::vector<std::byte> encoded = EncodeTokenPayload(slots);
    ASSERT_EQ(encoded.size(), slots.size() * 4);
    EXPECT_EQ(DecodeTokenPayload(encoded), slots);

    // An empty slot list round-trips as an empty payload.
    const std::vector<std::byte> empty = EncodeTokenPayload({});
    EXPECT_TRUE(empty.empty());
    EXPECT_TRUE(DecodeTokenPayload(empty).empty());

    // Pure translation: a tampered slot byte decodes to the tampered value.
    std::vector<std::byte> tampered = encoded;
    tampered[0] = std::byte{9};
    EXPECT_EQ(DecodeTokenPayload(tampered), (std::vector<std::uint32_t>{9, 3, 7}));

    // One to three trailing bytes are ignored: the count is the payload
    // length divided by four, rounded down.
    tampered.resize(tampered.size() + 2, std::byte{0xFF});
    EXPECT_EQ(DecodeTokenPayload(tampered), (std::vector<std::uint32_t>{9, 3, 7}));

    // Four appended bytes read back as one more slot: the length is the count.
    std::vector<std::byte> extended = encoded;
    extended.resize(extended.size() + 4, std::byte{0});
    EXPECT_EQ(DecodeTokenPayload(extended), (std::vector<std::uint32_t>{0, 3, 7, 0}));
}

TEST(TokenCodecTest, PlanPayloadRoundsTripBatchIdAndSlots)
{
    // The receipt handle rides first as u64, then one u32 per slot.
    const PlanPayload plan{
        std::uint64_t{0x0102030405060708},
        {0, 3, 7}
    };
    const std::vector<std::byte> encoded = EncodePlanPayload(plan);
    ASSERT_EQ(encoded.size(), 8 + plan.slots.size() * 4);
    const PlanPayload decoded = DecodePlanPayload(encoded);
    EXPECT_EQ(decoded.batchId, plan.batchId);
    EXPECT_EQ(decoded.slots, plan.slots);
}

TEST(TokenCodecTest, PlanPayloadEmptyIsNoPlan)
{
    // An idle plan encodes to the bare 8-byte handle; batchId 0 is the
    // reserved no-plan handle, so the round trip is the no-plan payload.
    const PlanPayload none;
    const std::vector<std::byte> encoded = EncodePlanPayload(none);
    ASSERT_EQ(encoded.size(), std::size_t{8});
    const PlanPayload decoded = DecodePlanPayload(encoded);
    EXPECT_EQ(decoded.batchId, std::uint64_t{0});
    EXPECT_TRUE(decoded.slots.empty());
}

TEST(TokenCodecTest, PlanPayloadShortInputDecodesToNoPlan)
{
    // Anything shorter than the 8-byte handle head is no plan at all -- a
    // token payload of stray slots never impersonates a plan.
    for (std::size_t len = 0; len < 8; ++len) {
        const std::vector<std::byte> stray(len, std::byte{0xFF});
        const PlanPayload decoded = DecodePlanPayload(stray);
        EXPECT_EQ(decoded.batchId, std::uint64_t{0}) << "length " << len;
        EXPECT_TRUE(decoded.slots.empty()) << "length " << len;
    }
}

TEST(TokenCodecTest, PlanPayloadIgnoresTrailingBytes)
{
    // One to three trailing bytes past the slots are ignored: the count is
    // the remaining length divided by four, rounded down.
    const PlanPayload plan{
        std::uint64_t{7},
        {11, 12}
    };
    std::vector<std::byte> encoded = EncodePlanPayload(plan);
    encoded.resize(encoded.size() + 3, std::byte{0xFF});
    const PlanPayload decoded = DecodePlanPayload(encoded);
    EXPECT_EQ(decoded.batchId, plan.batchId);
    EXPECT_EQ(decoded.slots, plan.slots);

    // Four appended bytes read back as one more slot.
    std::vector<std::byte> extended = EncodePlanPayload(plan);
    extended.resize(extended.size() + 4, std::byte{0});
    const PlanPayload extendedDecoded = DecodePlanPayload(extended);
    EXPECT_EQ(extendedDecoded.slots, (std::vector<std::uint32_t>{11, 12, 0}));
}

}  // namespace
}  // namespace UC::Store
