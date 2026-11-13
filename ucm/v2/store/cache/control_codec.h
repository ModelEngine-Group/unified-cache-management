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
#include <string>
#include <vector>
#include "status/status.h"
#include "type/block.h"

namespace UC::Store {

/** Control-plane opcode, mapped one-to-one onto the Store admission protocol. */
enum class ControlOp : std::uint8_t {
    Hello = 0, /** Handshake */
    Lookup = 1,
    Touch = 2,
    Pin = 3,
    Unpin = 4,
    Alloc = 5,
    Commit = 6,
};

/**
 * @brief One control-plane request (any op except Hello). Wire layout,
 *        little-endian:
 *        u8 op | u32 seq | u32 count | BlockId[count] (16B each) | u8 hasTtl | i64 ttlMs | u8
 * success
 */
struct ControlRequest {
    ControlOp op{ControlOp::Lookup};
    std::uint32_t seq{0};
    std::vector<BlockId> blockIds;
    /** Whether ttlMs carries a TTL fact (RequestContext.ttl); false leaves ttlMs ignored. */
    bool hasTtl{false};
    std::int64_t ttlMs{0};
    /** Commit's success flag; ignored by the other ops. */
    bool success{false};
};

/**
 * @brief One control-plane response. Wire layout, little-endian:
 *        u32 seq | u8 ok | i32 statusCode | u32 msgLen | msg |
 *        u8 resultKind (0 none, 1 hits, 2 slots) | [result body] | u32 tokenLen | token bytes
 *
 * The result body is single-choice by op: Lookup answers with hits, Pin/Alloc
 * with slots; every other op (and any failure) carries none. A successful
 * Pin/Alloc additionally attaches the minted AddressedToken payload
 * (self-addressing envelope) as opaque bytes -- its internal encoding belongs
 * to shm_layout.
 *
 * Handshake replies reuse this frame: seq stays 0. A
 * successful Hello answers ok=1 with no result body, the shared memfd riding
 * out-of-band over SCM_RIGHTS for data-plane roles only; a rejected Hello
 * answers ok=0 with the statusCode and message telling the joiner why.
 */
struct ControlResponse {
    std::uint32_t seq{0};
    bool ok{false};
    std::int32_t statusCode{0};
    std::string message;
    /** Lookup result: hits[i] != 0 iff blockIds[i] is present. */
    std::vector<std::uint8_t> hits;
    /** Pin/Alloc result: slot index per block. */
    std::vector<std::uint32_t> slots;
    /** AddressedToken payload of a successful Pin/Alloc; empty otherwise. */
    std::vector<std::byte> tokenPayload;
};

/**
 * @brief Handshake request sent by a joining process.
 *        Wire layout: u8 op(=Hello) | u32 seq | role | uniqueId |
 *        u64 blockTotalNumber | u64 blockSizeBytes | replacementPolicy | u8 forceHugePage,
 *        where every string is a u16 length prefix followed by raw bytes.
 *        The reply is a plain ControlResponse with seq=0 -- see the handshake
 *        note on ControlResponse.
 */
struct HelloRequest {
    std::string role;
    std::string uniqueId;
    std::uint64_t blockTotalNumber{0};
    std::uint64_t blockSizeBytes{0};
    std::string replacementPolicy;
    bool forceHugePage{false};
};

/**
 * @brief Encodes a data-plane request (Lookup/Touch/Pin/Unpin/Alloc/Commit).
 *
 * @return Expected the wire bytes on success; InvalidParam when op is Hello or
 *         out of range.
 */
[[nodiscard]] Expected<std::vector<std::byte>> EncodeRequest(const ControlRequest& request);

/**
 * @brief Decodes a data-plane request; rejects Hello bodies, truncated or
 *        over-long inputs, and unknown opcodes with InvalidParam.
 */
[[nodiscard]] Expected<ControlRequest> DecodeRequest(const std::vector<std::byte>& data);

/**
 * @brief Encodes a response. InvalidParam when both hits and slots are set
 *         (the result body is single-choice).
 */
[[nodiscard]] Expected<std::vector<std::byte>> EncodeResponse(const ControlResponse& response);

/**
 * @brief Decodes a response; fails with InvalidParam on truncation, trailing
 *        bytes, or an unknown result kind.
 */
[[nodiscard]] Expected<ControlResponse> DecodeResponse(const std::vector<std::byte>& data);

/** @brief Encodes a handshake request; role/uniqueId/policy are limited to 65535 bytes each. */
[[nodiscard]] Expected<std::vector<std::byte>> EncodeHelloRequest(const HelloRequest& hello);

/** @brief Decodes a handshake request; rejects non-Hello bodies with InvalidParam. */
[[nodiscard]] Expected<HelloRequest> DecodeHelloRequest(const std::vector<std::byte>& data);

}  // namespace UC::Store
