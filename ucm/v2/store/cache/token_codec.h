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
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace UC::Store {

/**
 * @brief Payload of an AddressedToken minted over the shared pool: a bare
 *        little-endian u32 slot array. Pure translation with no policy --
 *        the recipient resolves slots through its own geometry, and slot
 *        validity is the minter's contract (see SlotAddress's precondition).
 */
[[nodiscard]] std::vector<std::byte> EncodeTokenPayload(const std::vector<std::uint32_t>& slots);

/**
 * @brief Decodes a token payload: slot count is the payload byte length
 *        divided by four, rounded down; one to three trailing bytes are
 *        ignored. No semantic checks -- a payload that did not come from
 *        EncodeTokenPayload is out of contract.
 */
[[nodiscard]] std::vector<std::uint32_t> DecodeTokenPayload(const std::vector<std::byte>& data);

/**
 * @brief Body of one lower-tier transfer plan: the receipt handle minted
 *        with the plan, plus the slots the plan covers. The slots are
 *        shared-pool slots -- offload sources or prefetch destinations;
 *        which of the two the plan is belongs to the control message
 *        carrying it, not to the payload.
 */
struct PlanPayload {
    /**
     * @brief Receipt handle correlating the plan across its process round
     *        trip -- minted where the plan is minted, echoed by whoever
     *        reports on the plan. 0 is reserved and never minted: a payload
     *        carrying it is the explicit no-plan sentinel.
     */
    std::uint64_t batchId{0};
    /** @brief One slot per planned key, in the order the keys ride. */
    std::vector<std::uint32_t> slots;
};

/**
 * @brief Encodes a plan payload: little-endian u64 batchId, then one
 *        little-endian u32 per slot. Pure translation with no policy --
 *        slot validity is the minter's contract (see SlotAddress's
 *        precondition).
 */
[[nodiscard]] std::vector<std::byte> EncodePlanPayload(const PlanPayload& plan);

/**
 * @brief Decodes a plan payload: fewer than eight bytes decode to the
 *        no-plan payload (batchId 0, no slots); past the batchId, the slot
 *        count is the remaining byte length divided by four, rounded down,
 *        so one to three trailing bytes are ignored. No semantic checks --
 *        a payload that did not come from EncodePlanPayload is out of
 *        contract.
 */
[[nodiscard]] PlanPayload DecodePlanPayload(const std::vector<std::byte>& data);

}  // namespace UC::Store
