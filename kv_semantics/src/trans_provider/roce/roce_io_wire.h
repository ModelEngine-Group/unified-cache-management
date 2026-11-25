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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include "kv_types.h"

// Data path wire format of the CPU-over-RoCE provider.
//
// The client packs an SQE (the structure kv_protocol describes) into a send
// buffer and hands it to the provider together with the flag buffer the caller
// polls for the completion.  The CPU path sends that SQE with a two sided RDMA
// send and the store answers with a two sided send carrying the completion
// record, so the provider has to know three things about the payloads:
//
//   - which cid a packed SQE belongs to, so an answer can be matched to it;
//   - how many bytes of the completion the caller's flag buffer holds, so a
//     response can be copied without a byte past its slot;
//   - how a completion record looks, which is four header dwords (the third one
//     carries the cid and the CQE status) followed by the result buffer.
//
// Completion records are what the store appliance sends: the layout is the
// recv_data_block of the NDS protocol, which the same appliance speaks.  A
// response therefore never exceeds kResponseMaxBytes, and a provider that keeps
// that much room per posted receive buffer accepts every answer.

namespace kv {
namespace roce {

// Header of a completion record: dword 0..2 carry per opcode data, dword 3 the
// cid in the low half and the fifteen bit CQE status above it.
constexpr std::size_t kResponseHeaderBytes = 16;
// Largest result buffer the appliance puts behind the header, in bytes.
constexpr std::size_t kResponseResultBytes = 64;
constexpr std::size_t kResponseMaxBytes = kResponseHeaderBytes + kResponseResultBytes;

// An SQE starts with sixteen header dwords; batch entry and key entry records
// follow it.
constexpr std::size_t kSqeHeaderDwords = 16;
constexpr std::size_t kSqeBatchEntryDwords = 9;
constexpr std::size_t kSqeKeyEntryDwords = 4;
constexpr std::size_t kSqeDwordBytes = sizeof(std::uint32_t);
// Largest SQE the protocol can pack: an exist of 256 keys (16 + 256 * 4 dwords).
constexpr std::size_t kSqeMaxBytes =
    (kSqeHeaderDwords + 256 * kSqeKeyEntryDwords) * kSqeDwordBytes;

// Header bits of an SQE.  The rflag asks the store to write the completion into
// the response buffer the request names; the send/response mode clears it and
// the store answers with a send instead.
constexpr std::uint32_t kSqeFixedBits = 0x1U;
constexpr std::uint32_t kSqeRflagBit = 13U;
constexpr std::uint32_t kSqeOpcodeMask = 0xFFU;
constexpr std::uint32_t kSqeCidShift = 16U;
constexpr std::uint32_t kSqeCidMask = 0xFFFFU;

// Where the completion record keeps the cid and the status.
constexpr std::uint32_t kCqeStatusShift = 17U;
constexpr std::uint32_t kCqeStatusMask = 0x7FFFU;

// CQE status the provider itself reports when the verbs path failed before the
// store could answer.  The low fifteen bits travel on the wire; the client adds
// the 0x10000 base (StatusCode::CQE_INTERNAL_ERROR).
constexpr std::uint16_t kCqeStatusInternalError = 0x006U;

constexpr std::uint8_t kOpcodeStore = 0x1U;
constexpr std::uint8_t kOpcodeRetrieve = 0x2U;
constexpr std::uint8_t kOpcodeBatchStore = 0x45U;
constexpr std::uint8_t kOpcodeBatchRetrieve = 0x46U;
constexpr std::uint8_t kOpcodeDelete = 0x8U;
constexpr std::uint8_t kOpcodeExist = 0xCU;
constexpr std::uint8_t kOpcodeKeepAlive = 0xF4U;

// Batch limits the packed SQE format (kv_protocol) accepts.
constexpr std::uint16_t kMaxBatchStoreKeys = 110;
constexpr std::uint16_t kMaxDeleteKeys = 254;
constexpr std::uint16_t kMaxExistKeys = 256;

// What the provider needs to know about one packed request.
struct RequestDescription {
    std::uint16_t cid{0};
    std::uint8_t opcode{0};
    std::uint16_t batchNumber{1};
    // Bytes of the completion record the caller's flag buffer holds.  It is the
    // same size the client reserved (see sqe_request.cc), so a response written
    // through it stays inside the slot.
    std::size_t responseBytes{0};
    // Size the packed SQE must have.
    std::size_t sqeBytes{0};
    // True when the request asks the store to write the completion into the
    // response buffer named by the SQE (rflag set).  The send/response path
    // needs it clear; the flags are reported so the caller can reject a request
    // that would never produce a send.
    bool writeback{false};
};

namespace detail {

inline bool IsBatchOpcode(std::uint8_t opcode)
{
    return opcode == kOpcodeBatchStore || opcode == kOpcodeBatchRetrieve;
}

inline bool IsKeyOpcode(std::uint8_t opcode)
{
    return opcode == kOpcodeDelete || opcode == kOpcodeExist;
}

}  // namespace detail

// Bytes of the completion record a request of this opcode and batch size gets
// room for.  Mirrors GetFlagBufferSize() of the client side: a batch store or
// retrieve reports four bits per key, delete and exist one bit, and the single
// key opcodes carry no per entry result.
inline std::size_t ResponseBytesFor(std::uint8_t opcode, std::uint16_t batchNumber)
{
    if (detail::IsBatchOpcode(opcode)) {
        return kResponseHeaderBytes + static_cast<std::size_t>(batchNumber + 1U) / 2U;
    }
    if (detail::IsKeyOpcode(opcode)) {
        return kResponseHeaderBytes + static_cast<std::size_t>(batchNumber + 7U) / 8U;
    }
    if (opcode == kOpcodeStore || opcode == kOpcodeRetrieve || opcode == kOpcodeKeepAlive) {
        return kResponseHeaderBytes + 1U;
    }
    return 0;
}

// Reads cid, opcode and batch size out of a packed SQE and checks that the
// buffer actually holds one.  The caller packs with ProtocolManager, so a
// mismatch here means the wrong buffer was handed to Send(), not a peer error.
inline Status DescribeRequest(const void* sqe, std::size_t length,
                              RequestDescription& description)
{
    description = RequestDescription{};
    if (sqe == nullptr) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "send buffer is null");
    }
    if (reinterpret_cast<std::uintptr_t>(sqe) % sizeof(std::uint32_t) != 0U) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "send buffer is not aligned to a 32 bit word");
    }
    if (length < kSqeHeaderDwords * sizeof(std::uint32_t)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "send buffer of " + std::to_string(length) +
                                 " bytes is shorter than an SQE header");
    }
    if (length > kSqeMaxBytes) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "send buffer of " + std::to_string(length) +
                                 " bytes exceeds the largest SQE of " +
                                 std::to_string(kSqeMaxBytes) + " bytes");
    }

    const auto* dwords = static_cast<const std::uint32_t*>(sqe);
    const std::uint32_t header = dwords[0];
    const auto opcode = static_cast<std::uint8_t>(header & kSqeOpcodeMask);
    // Every opcode but the keep alive one carries the fixed bits above the opcode;
    // the keep alive format reserves those two bits as zero.
    if (opcode != kOpcodeKeepAlive && ((header >> 14U) & 0x3U) != kSqeFixedBits) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "send buffer does not start with fixed SQE bits");
    }
    const std::uint16_t cid =
        static_cast<std::uint16_t>((header >> kSqeCidShift) & kSqeCidMask);
    if (cid == 0U) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "SQE cid is zero");
    }

    RequestDescription decoded;
    decoded.cid = cid;
    decoded.writeback = ((header >> kSqeRflagBit) & 0x1U) != 0U;
    decoded.opcode = opcode;
    if (detail::IsBatchOpcode(opcode) || detail::IsKeyOpcode(opcode)) {
        const auto batchNumber = static_cast<std::uint16_t>(dwords[10] & 0xFFFFU);
        if (batchNumber == 0U) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "SQE batch number is zero");
        }
        std::uint16_t maximum = kMaxBatchStoreKeys;
        std::size_t entryDwords = kSqeBatchEntryDwords;
        if (detail::IsKeyOpcode(opcode)) {
            maximum = opcode == kOpcodeDelete ? kMaxDeleteKeys : kMaxExistKeys;
            entryDwords = kSqeKeyEntryDwords;
        }
        if (batchNumber > maximum) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "SQE batch number " + std::to_string(batchNumber) +
                                     " exceeds the opcode limit of " + std::to_string(maximum));
        }
        decoded.batchNumber = batchNumber;
        decoded.sqeBytes = (kSqeHeaderDwords + batchNumber * entryDwords) * sizeof(std::uint32_t);
    } else if (opcode == kOpcodeStore || opcode == kOpcodeRetrieve ||
               opcode == kOpcodeKeepAlive) {
        decoded.batchNumber = 1;
        decoded.sqeBytes = kSqeHeaderDwords * sizeof(std::uint32_t);
    } else {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "SQE opcode " + std::to_string(static_cast<unsigned>(opcode)) +
                                 " is not supported by the send/response path");
    }

    if (length != decoded.sqeBytes) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "send buffer of " + std::to_string(length) +
                                 " bytes does not match the SQE size of " +
                                 std::to_string(decoded.sqeBytes) + " bytes");
    }
    decoded.responseBytes = ResponseBytesFor(opcode, decoded.batchNumber);
    description = decoded;
    return Status::OK();
}

// Copies a received completion record into the caller's flag buffer.  The cid
// and status dword is stored last with release semantics because that is what
// the caller polls (ProtocolManager::PollResponseCid() reads it with acquire):
// until it is published the buffer still looks empty.
inline Status PublishResponse(const void* response, std::size_t receivedBytes, void* flagBuffer,
                              std::size_t capacity)
{
    if (response == nullptr || flagBuffer == nullptr) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "response or flag buffer is null");
    }
    if (receivedBytes < kResponseHeaderBytes) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "response of " + std::to_string(receivedBytes) +
                                 " bytes carries no CQE header");
    }
    if (capacity < kResponseHeaderBytes) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "flag buffer of " + std::to_string(capacity) +
                                 " bytes cannot hold a CQE header");
    }

    const std::size_t room = capacity - kResponseHeaderBytes;
    const std::size_t available = receivedBytes - kResponseHeaderBytes;
    const std::size_t payloadBytes = std::min(room, available);

    const auto* source = static_cast<const std::uint8_t*>(response);
    auto* target = static_cast<std::uint8_t*>(flagBuffer);
    auto* targetDwords = static_cast<std::uint32_t*>(flagBuffer);

    targetDwords[0] = static_cast<const std::uint32_t*>(response)[0];
    targetDwords[1] = static_cast<const std::uint32_t*>(response)[1];
    targetDwords[2] = static_cast<const std::uint32_t*>(response)[2];
    if (payloadBytes > 0U) {
        std::memcpy(target + kResponseHeaderBytes, source + kResponseHeaderBytes, payloadBytes);
    }
    const std::uint32_t statusDword = static_cast<const std::uint32_t*>(response)[3];
    __atomic_store_n(targetDwords + 3, statusDword, __ATOMIC_RELEASE);
    return Status::OK();
}

}  // namespace roce
}  // namespace kv
