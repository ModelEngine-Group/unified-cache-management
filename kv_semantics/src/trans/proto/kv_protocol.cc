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
#include <cstring>

namespace kv {

static void UnpackCqeBase(const std::uint32_t* data, KvResponse& out)
{
    out.cid = data[3] & 0xFFFF;
    out.status = (data[3] >> 17) & 0x7FFF;
}

static void UnpackResultBuffer4Bit(const std::uint32_t* result_data, std::uint16_t batch_number,
                                   std::vector<std::uint8_t>& result_buffer)
{
    result_buffer.resize(batch_number);
    for (std::uint16_t i = 0; i < batch_number; ++i) {
        std::uint32_t dword_idx = i / 8;
        std::uint32_t slot = (i % 8) * 4;
        result_buffer[i] = (result_data[dword_idx] >> slot) & 0xF;
    }
}

static void UnpackResultBuffer1Bit(const std::uint32_t* result_data, std::uint16_t batch_number,
                                   std::vector<std::uint8_t>& result_buffer)
{
    result_buffer.resize(batch_number);
    for (std::uint16_t i = 0; i < batch_number; ++i) {
        std::uint32_t dword_idx = i / 32;
        std::uint32_t bit = i % 32;
        result_buffer[i] = (result_data[dword_idx] >> bit) & 0x1;
    }
}

static Status VerifyFixedBits(const std::uint32_t* data, const char* log_prefix)
{
    if (((data[0] >> 14) & 0x3) != kFixedBits) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             std::string(log_prefix) + ": fixed bits mismatch");
    }
    return Status::OK();
}

static Status VerifyRflagConsistency(const std::uint32_t* data, const char* log_prefix)
{
    bool rflag = (data[0] >> 13) & 0x1;
    std::uint64_t resp_addr =
        static_cast<std::uint64_t>(data[3]) | (static_cast<std::uint64_t>(data[4]) << 32);
    std::uint32_t resp_mr_key = data[5];
    if (rflag) {
        if (resp_addr == 0) {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                std::string(log_prefix) + ": rflag set but response_buffer_addr is zero");
        }
        if (resp_mr_key == 0) {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                std::string(log_prefix) + ": rflag set but response_mr_key is zero");
        }
    } else {
        if (resp_addr != 0) {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                std::string(log_prefix) + ": rflag false but response_buffer_addr non-zero");
        }
        if (resp_mr_key != 0) {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                std::string(log_prefix) + ": rflag false but response_mr_key non-zero");
        }
    }
    return Status::OK();
}

static Status VerifyCacheKeyEntry(const std::uint32_t* key_dwords, const std::string& log_prefix)
{
    const auto* bytes = reinterpret_cast<const std::byte*>(key_dwords);
    const auto keyAllZero = std::all_of(bytes, bytes + kCacheKeySizeBytes,
                                        [](std::byte value) { return value == std::byte{0}; });
    if (keyAllZero) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, log_prefix + ": key is all zeros");
    }
    const auto tailAllZero = std::all_of(bytes + kCacheKeySizeBytes, bytes + kKeyEntrySizeBytes,
                                         [](std::byte value) { return value == std::byte{0}; });
    if (!tailAllZero) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             log_prefix + ": reserved key bytes are non-zero");
    }
    return Status::OK();
}

static bool IsCacheKeyAllZero(const CacheKey& key)
{
    return std::all_of(key.begin(), key.end(),
                       [](std::byte value) { return value == std::byte{0}; });
}

static Status VerifyBatchEntry(const std::uint32_t* base, const std::string& log_prefix,
                               std::uint16_t i)
{
    if (base[0] % kAlignmentBytes != 0) {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            log_prefix + ": entry[" + std::to_string(i) + "] offset not 512B aligned");
    }

    auto key_status =
        VerifyCacheKeyEntry(&base[1], log_prefix + ": entry[" + std::to_string(i) + "]");
    if (!key_status.ok()) { return key_status; }

    std::uint64_t entry_addr =
        static_cast<std::uint64_t>(base[5]) | (static_cast<std::uint64_t>(base[6]) << 32);
    if (entry_addr == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             log_prefix + ": entry[" + std::to_string(i) + "] buffer_addr is zero");
    }

    std::uint32_t entry_length = base[7] & 0xFFFFFF;
    if (entry_length == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             log_prefix + ": entry[" + std::to_string(i) + "] length is zero");
    }
    if (entry_length % kAlignmentBytes != 0) {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            log_prefix + ": entry[" + std::to_string(i) + "] length not 512B aligned");
    }

    if ((base[8] >> 24) != static_cast<std::uint32_t>(DptrType::Standard)) {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            log_prefix + ": entry[" + std::to_string(i) + "] DptrType != Standard");
    }

    return Status::OK();
}

Status KvStoreProtocol::PackSqe(const SqeRequest& req, std::uint32_t* target)
{
    auto& r = static_cast<const KvStoreRequest&>(req);
    auto status = ValidateRequest(r);
    if (!status.ok()) { return status; }

    target[0] = (r.cid << 16) | (kFixedBits << 14) | static_cast<std::uint32_t>(KvOpcode::Store);
    target[1] = r.kv_ns_id;
    target[2] = ((r.dtype & 0x7) << 13) | ((r.dspec & 0x1F) << 8);
    target[6] = r.buffer_addr & 0xFFFFFFFFULL;
    target[7] = (r.buffer_addr >> 32) & 0xFFFFFFFFULL;
    target[8] = ((r.mr_key & 0xFF) << 24) | (r.buffer_length & 0xFFFFFF);
    target[9] =
        (static_cast<std::uint32_t>(DptrType::Standard) << 24) | ((r.mr_key >> 8) & 0xFFFFFF);
    target[10] = r.offset;
    target[11] = (r.lr ? (1U << 31) : 0) | (r.length & 0xFFFFFF);

    std::memcpy(&target[12], r.key.data(), kCacheKeySizeBytes);
    return Status::OK();
}

Status KvStoreProtocol::ValidateRequest(const KvStoreRequest& r) const
{
    if (r.buffer_addr == 0) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "buffer_addr is zero in Store request");
    }
    if (r.buffer_length == 0) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "buffer_length is zero in Store request");
    }
    if (r.buffer_length % kAlignmentBytes != 0) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "buffer_length(" + std::to_string(r.buffer_length) +
                                 ") must be 512B aligned in Store request");
    }
    if (r.buffer_length > 0xFFFFFF) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "buffer_length(" + std::to_string(r.buffer_length) +
                                 ") exceeds 24-bit limit in Store request");
    }
    if (r.offset % kAlignmentBytes != 0) [[unlikely]] {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "offset(" + std::to_string(r.offset) + ") must be 512B aligned in Store request");
    }
    if (r.length == 0) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "length is 1-based, must be non-zero in Store request");
    }
    if (r.length > 0xFFFFFF) [[unlikely]] {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "length(" + std::to_string(r.length) + ") exceeds 24-bit limit in Store request");
    }
    if (IsCacheKeyAllZero(r.key)) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "key is all zeros in Store request");
    }
    if (r.dtype > 0x7) [[unlikely]] {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "dtype(" + std::to_string(r.dtype) + ") exceeds 3-bit limit in Store request");
    }
    if (r.dspec > 0x1F) [[unlikely]] {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "dspec(" + std::to_string(r.dspec) + ") exceeds 5-bit limit in Store request");
    }
    return Status::OK();
}

Status KvStoreProtocol::UnpackCqe(const std::uint32_t* data, std::uint16_t, KvResponse& out) const
{
    UnpackCqeBase(data, out);
    return Status::OK();
}

Status KvStoreProtocol::VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const
{
    constexpr std::size_t kExpectedLength = kSqeDwordCount * sizeof(std::uint32_t);
    if (length != kExpectedLength) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Store: length(" + std::to_string(length) + ") != expected(" +
                                 std::to_string(kExpectedLength) + ")");
    }

    auto status = VerifyFixedBits(data, "Store");
    if (!status.ok()) { return status; }
    if ((data[0] >> 8) & 0x3F) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Store: reserved bits non-zero in data[0] bit8-13");
    }

    if (data[2] & 0xFFFF00FFU) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Store: reserved bits non-zero in data[2] bit0-7,bit16-31");
    }

    if (data[3] || data[4] || data[5]) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Store: reserved bits non-zero in data[3-5]");
    }

    std::uint64_t buffer_addr =
        static_cast<std::uint64_t>(data[6]) | (static_cast<std::uint64_t>(data[7]) << 32);
    if (buffer_addr == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Store: buffer_addr is zero");
    }

    std::uint32_t buffer_length = data[8] & 0xFFFFFF;
    if (buffer_length == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Store: buffer_length is zero");
    }
    if (buffer_length % kAlignmentBytes != 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Store: buffer_length not 512B aligned");
    }

    if ((data[9] >> 24) != static_cast<std::uint32_t>(DptrType::Standard)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Store: DptrType != Standard");
    }

    std::uint32_t offset = data[10];
    if (offset % kAlignmentBytes != 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Store: offset not 512B aligned");
    }

    std::uint32_t kv_length = data[11] & 0xFFFFFF;
    if (kv_length == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Store: length is zero");
    }
    if (data[11] & 0x7F000000U) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Store: reserved bits non-zero in data[11] bit24-30");
    }

    status = VerifyCacheKeyEntry(&data[12], "Store");
    if (!status.ok()) { return status; }

    return Status::OK();
}

Status KvRetrieveProtocol::PackSqe(const SqeRequest& req, std::uint32_t* target)
{
    auto& r = static_cast<const KvRetrieveRequest&>(req);
    auto status = ValidateRequest(r);
    if (!status.ok()) { return status; }

    target[0] = (r.cid << 16) | (kFixedBits << 14) | static_cast<std::uint32_t>(KvOpcode::Retrieve);
    target[1] = r.kv_ns_id;
    target[6] = r.buffer_addr & 0xFFFFFFFFULL;
    target[7] = (r.buffer_addr >> 32) & 0xFFFFFFFFULL;
    target[8] = ((r.mr_key & 0xFF) << 24) | (r.buffer_length & 0xFFFFFF);
    target[9] =
        (static_cast<std::uint32_t>(DptrType::Standard) << 24) | ((r.mr_key >> 8) & 0xFFFFFF);
    target[10] = r.offset;
    target[11] = (r.lr ? (1U << 31) : 0) | (r.length & 0xFFFFFF);

    std::memcpy(&target[12], r.key.data(), kCacheKeySizeBytes);
    return Status::OK();
}

Status KvRetrieveProtocol::ValidateRequest(const KvRetrieveRequest& r) const
{
    if (r.buffer_addr == 0) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "buffer_addr is zero in Retrieve request");
    }
    if (r.buffer_length == 0) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "buffer_length is zero in Retrieve request");
    }
    if (r.buffer_length % kAlignmentBytes != 0) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "buffer_length(" + std::to_string(r.buffer_length) +
                                 ") must be 512B aligned in Retrieve request");
    }
    if (r.buffer_length > 0xFFFFFF) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "buffer_length(" + std::to_string(r.buffer_length) +
                                 ") exceeds 24-bit limit in Retrieve request");
    }
    if (r.offset % kAlignmentBytes != 0) [[unlikely]] {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "offset(" + std::to_string(r.offset) + ") must be 512B aligned in Retrieve request");
    }
    if (r.length == 0) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "length is 1-based, must be non-zero in Retrieve request");
    }
    if (r.length > 0xFFFFFF) [[unlikely]] {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "length(" + std::to_string(r.length) + ") exceeds 24-bit limit in Retrieve request");
    }
    if (IsCacheKeyAllZero(r.key)) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "key is all zeros in Retrieve request");
    }
    return Status::OK();
}

Status KvRetrieveProtocol::UnpackCqe(const std::uint32_t* data, std::uint16_t,
                                     KvResponse& out) const
{
    UnpackCqeBase(data, out);
    return Status::OK();
}

Status KvRetrieveProtocol::VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const
{
    constexpr std::size_t kExpectedLength = kSqeDwordCount * sizeof(std::uint32_t);
    if (length != kExpectedLength) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Retrieve: length(" + std::to_string(length) + ") != expected(" +
                                 std::to_string(kExpectedLength) + ")");
    }

    auto status = VerifyFixedBits(data, "Retrieve");
    if (!status.ok()) { return status; }
    if ((data[0] >> 8) & 0x3F) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Retrieve: reserved bits non-zero in data[0] bit8-13");
    }

    if (data[2] != 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Retrieve: reserved bits non-zero in data[2]");
    }

    if (data[3] || data[4] || data[5]) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Retrieve: reserved bits non-zero in data[3-5]");
    }

    std::uint64_t buffer_addr =
        static_cast<std::uint64_t>(data[6]) | (static_cast<std::uint64_t>(data[7]) << 32);
    if (buffer_addr == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Retrieve: buffer_addr is zero");
    }

    std::uint32_t buffer_length = data[8] & 0xFFFFFF;
    if (buffer_length == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Retrieve: buffer_length is zero");
    }
    if (buffer_length % kAlignmentBytes != 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Retrieve: buffer_length not 512B aligned");
    }

    if ((data[9] >> 24) != static_cast<std::uint32_t>(DptrType::Standard)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Retrieve: DptrType != Standard");
    }

    std::uint32_t offset = data[10];
    if (offset % kAlignmentBytes != 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Retrieve: offset not 512B aligned");
    }

    std::uint32_t kv_length = data[11] & 0xFFFFFF;
    if (kv_length == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Retrieve: length is zero");
    }
    if (data[11] & 0x7F000000U) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Retrieve: reserved bits non-zero in data[11] bit24-30");
    }

    status = VerifyCacheKeyEntry(&data[12], "Retrieve");
    if (!status.ok()) { return status; }

    return Status::OK();
}

std::size_t KvBatchStoreProtocol::PackedSize(const SqeRequest& req) const
{
    auto& r = static_cast<const KvBatchStoreRequest&>(req);
    return (kSqeDwordCount + r.batch_number * kBatchEntryDwordCount) * sizeof(std::uint32_t);
}

Status KvBatchStoreProtocol::PackSqe(const SqeRequest& req, std::uint32_t* target)
{
    auto& r = static_cast<const KvBatchStoreRequest&>(req);
    auto status = ValidateRequest(r);
    if (!status.ok()) { return status; }

    target[0] =
        (r.cid << 16) | (kFixedBits << 14) | static_cast<std::uint32_t>(KvOpcode::BatchStore);
    if (r.rflag) { target[0] |= (1U << 13); }
    target[1] = r.kv_ns_id;
    target[2] = ((r.dtype & 0x7) << 13) | ((r.dspec & 0x1F) << 8);
    target[3] = r.response_buffer_addr & 0xFFFFFFFFULL;
    target[4] = (r.response_buffer_addr >> 32) & 0xFFFFFFFFULL;
    target[5] = r.response_mr_key;
    target[8] = r.batch_number * kBatchEntrySizeBytes;
    target[9] = static_cast<std::uint32_t>(DptrType::Batch) << 24;
    target[10] = r.batch_number;
    if (r.lr) { target[11] |= (1U << 31); }

    for (std::size_t i = 0; i < r.batch_number; ++i) {
        PackEntry(r.entries[i], target + kSqeDwordCount + i * kBatchEntryDwordCount);
    }
    return Status::OK();
}

Status KvBatchStoreProtocol::ValidateRequest(const KvBatchStoreRequest& r) const
{
    if (r.batch_number == 0 || r.batch_number > kMaxBatchNumber) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "batch_number(" + std::to_string(r.batch_number) +
                                 ") must be in range [1, 110] in BatchStore request");
    }
    if (r.batch_number != r.entries.size()) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "batch_number(" + std::to_string(r.batch_number) +
                                 ") must equal entries.size()(" + std::to_string(r.entries.size()) +
                                 ") in BatchStore request");
    }
    if (r.rflag) {
        if (r.response_buffer_addr == 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "response_buffer_addr is zero in BatchStore request");
        }
        if (r.response_mr_key == 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "response_mr_key is zero in BatchStore request");
        }
    } else {
        if (r.response_buffer_addr != 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "response_buffer_addr must be zero when rflag is false in BatchStore request");
        }
        if (r.response_mr_key != 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "response_mr_key must be zero when rflag is false in BatchStore request");
        }
    }
    if (r.dtype > 0x7) [[unlikely]] {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "dtype(" + std::to_string(r.dtype) + ") exceeds 3-bit limit in BatchStore request");
    }
    if (r.dspec > 0x1F) [[unlikely]] {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "dspec(" + std::to_string(r.dspec) + ") exceeds 5-bit limit in BatchStore request");
    }
    for (std::size_t i = 0; i < r.batch_number; ++i) {
        const auto& entry = r.entries[i];
        if (entry.offset % kAlignmentBytes != 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "entry[" + std::to_string(i) + "] offset(" +
                                     std::to_string(entry.offset) +
                                     ") must be 512B aligned in BatchStore request");
        }
        if (IsCacheKeyAllZero(entry.key)) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "entry[" + std::to_string(i) + "] key is all zeros in BatchStore request");
        }
        if (entry.buffer_addr == 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "entry[" + std::to_string(i) + "] buffer_addr is zero in BatchStore request");
        }
        if (entry.length == 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "entry[" + std::to_string(i) + "] length is zero in BatchStore request");
        }
        if (entry.length % kAlignmentBytes != 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "entry[" + std::to_string(i) + "] length(" +
                                     std::to_string(entry.length) +
                                     ") must be 512B aligned in BatchStore request");
        }
        if (entry.length > 0xFFFFFF) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "entry[" + std::to_string(i) + "] length(" +
                                     std::to_string(entry.length) +
                                     ") exceeds 24-bit limit in BatchStore request");
        }
    }
    return Status::OK();
}

void KvBatchStoreProtocol::PackEntry(const KvBatchStoreEntry& entry, std::uint32_t* base)
{
    base[0] = entry.offset;

    std::memcpy(&base[1], entry.key.data(), kCacheKeySizeBytes);

    base[5] = entry.buffer_addr & 0xFFFFFFFFULL;
    base[6] = (entry.buffer_addr >> 32) & 0xFFFFFFFFULL;
    base[7] = ((entry.mr_key & 0xFF) << 24) | (entry.length & 0xFFFFFF);
    base[8] =
        (static_cast<std::uint32_t>(DptrType::Standard) << 24) | ((entry.mr_key >> 8) & 0xFFFFFF);
}

Status KvBatchStoreProtocol::UnpackCqe(const std::uint32_t* data, std::uint16_t batch_number,
                                       KvResponse& out) const
{
    UnpackCqeBase(data, out);
    UnpackResultBuffer4Bit(data + kCqeDwordCount, batch_number, out.result_buffer);
    return Status::OK();
}

Status KvBatchStoreProtocol::VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const
{
    auto status = VerifyFixedBits(data, "BatchStore");
    if (!status.ok()) { return status; }
    if ((data[0] >> 8) & 0x1F) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchStore: reserved bits non-zero in data[0] bit8-12");
    }

    if (data[2] & 0xFFFF00FFU) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchStore: reserved bits non-zero in data[2] bit0-7,bit16-31");
    }

    if (data[6] || data[7]) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchStore: reserved bits non-zero in data[6-7]");
    }

    status = VerifyRflagConsistency(data, "BatchStore");
    if (!status.ok()) { return status; }

    if (data[9] != (static_cast<std::uint32_t>(DptrType::Batch) << 24)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchStore: data[9] DptrType or reserved bits mismatch");
    }

    if (data[10] >> 16) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchStore: reserved bits non-zero in data[10] bit16-31");
    }
    std::uint16_t batch_number = data[10] & 0xFFFF;
    if (batch_number == 0 || batch_number > kMaxBatchNumber) {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "BatchStore: batch_number(" + std::to_string(batch_number) + ") out of range [1, 110]");
    }

    if (data[8] != static_cast<std::uint32_t>(batch_number) * kBatchEntrySizeBytes) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchStore: data[8](" + std::to_string(data[8]) +
                                 ") != batch_number * " + std::to_string(kBatchEntrySizeBytes));
    }

    std::size_t expected_length =
        (kSqeDwordCount + batch_number * kBatchEntryDwordCount) * sizeof(std::uint32_t);
    if (length != expected_length) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchStore: length(" + std::to_string(length) + ") != expected(" +
                                 std::to_string(expected_length) + ")");
    }

    if ((data[11] & 0x7FFFFFFFU) || data[12] || data[13] || data[14] || data[15]) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchStore: reserved bits non-zero in data[11-15]");
    }

    for (std::uint16_t i = 0; i < batch_number; ++i) {
        const std::uint32_t* base = data + kSqeDwordCount + i * kBatchEntryDwordCount;
        auto entry_status = VerifyBatchEntry(base, "BatchStore", i);
        if (!entry_status.ok()) { return entry_status; }
    }

    return Status::OK();
}

std::size_t KvBatchRetrieveProtocol::PackedSize(const SqeRequest& req) const
{
    auto& r = static_cast<const KvBatchRetrieveRequest&>(req);
    return (kSqeDwordCount + r.batch_number * kBatchEntryDwordCount) * sizeof(std::uint32_t);
}

Status KvBatchRetrieveProtocol::PackSqe(const SqeRequest& req, std::uint32_t* target)
{
    auto& r = static_cast<const KvBatchRetrieveRequest&>(req);
    auto status = ValidateRequest(r);
    if (!status.ok()) { return status; }

    target[0] =
        (r.cid << 16) | (kFixedBits << 14) | static_cast<std::uint32_t>(KvOpcode::BatchRetrieve);
    if (r.rflag) { target[0] |= (1U << 13); }
    target[1] = r.kv_ns_id;
    target[3] = r.response_buffer_addr & 0xFFFFFFFFULL;
    target[4] = (r.response_buffer_addr >> 32) & 0xFFFFFFFFULL;
    target[5] = r.response_mr_key;
    target[8] = r.batch_number * kBatchEntrySizeBytes;
    target[9] = static_cast<std::uint32_t>(DptrType::Batch) << 24;
    target[10] = r.batch_number;
    if (r.lr) { target[11] |= (1U << 31); }

    for (std::size_t i = 0; i < r.batch_number; ++i) {
        PackEntry(r.entries[i], target + kSqeDwordCount + i * kBatchEntryDwordCount);
    }
    return Status::OK();
}

Status KvBatchRetrieveProtocol::ValidateRequest(const KvBatchRetrieveRequest& r) const
{
    if (r.batch_number == 0 || r.batch_number > kMaxBatchNumber) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "batch_number(" + std::to_string(r.batch_number) +
                                 ") must be in range [1, 110] in BatchRetrieve request");
    }
    if (r.batch_number != r.entries.size()) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "batch_number(" + std::to_string(r.batch_number) +
                                 ") must equal entries.size()(" + std::to_string(r.entries.size()) +
                                 ") in BatchRetrieve request");
    }
    if (r.rflag) {
        if (r.response_buffer_addr == 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "response_buffer_addr is zero in BatchRetrieve request");
        }
        if (r.response_mr_key == 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "response_mr_key is zero in BatchRetrieve request");
        }
    } else {
        if (r.response_buffer_addr != 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "response_buffer_addr must be zero when rflag is false in BatchRetrieve request");
        }
        if (r.response_mr_key != 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "response_mr_key must be zero when rflag is false in BatchRetrieve request");
        }
    }
    for (std::size_t i = 0; i < r.batch_number; ++i) {
        const auto& entry = r.entries[i];
        if (entry.offset % kAlignmentBytes != 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "entry[" + std::to_string(i) + "] offset(" +
                                     std::to_string(entry.offset) +
                                     ") must be 512B aligned in BatchRetrieve request");
        }
        if (IsCacheKeyAllZero(entry.key)) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "entry[" + std::to_string(i) + "] key is all zeros in BatchRetrieve request");
        }
        if (entry.buffer_addr == 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "entry[" + std::to_string(i) + "] buffer_addr is zero in BatchRetrieve request");
        }
        if (entry.length == 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "entry[" + std::to_string(i) + "] length is zero in BatchRetrieve request");
        }
        if (entry.length % kAlignmentBytes != 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "entry[" + std::to_string(i) + "] length(" +
                                     std::to_string(entry.length) +
                                     ") must be 512B aligned in BatchRetrieve request");
        }
        if (entry.length > 0xFFFFFF) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "entry[" + std::to_string(i) + "] length(" +
                                     std::to_string(entry.length) +
                                     ") exceeds 24-bit limit in BatchRetrieve request");
        }
    }
    return Status::OK();
}

void KvBatchRetrieveProtocol::PackEntry(const KvBatchRetrieveEntry& entry, std::uint32_t* base)
{
    base[0] = entry.offset;

    std::memcpy(&base[1], entry.key.data(), kCacheKeySizeBytes);

    base[5] = entry.buffer_addr & 0xFFFFFFFFULL;
    base[6] = (entry.buffer_addr >> 32) & 0xFFFFFFFFULL;
    base[7] = ((entry.mr_key & 0xFF) << 24) | (entry.length & 0xFFFFFF);
    base[8] =
        (static_cast<std::uint32_t>(DptrType::Standard) << 24) | ((entry.mr_key >> 8) & 0xFFFFFF);
}

Status KvBatchRetrieveProtocol::UnpackCqe(const std::uint32_t* data, std::uint16_t batch_number,
                                          KvResponse& out) const
{
    UnpackCqeBase(data, out);
    UnpackResultBuffer4Bit(data + kCqeDwordCount, batch_number, out.result_buffer);
    return Status::OK();
}

Status KvBatchRetrieveProtocol::VerifyPackedBuffer(const std::uint32_t* data,
                                                   std::size_t length) const
{
    auto status = VerifyFixedBits(data, "BatchRetrieve");
    if (!status.ok()) { return status; }
    if ((data[0] >> 8) & 0x1F) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchRetrieve: reserved bits non-zero in data[0] bit8-12");
    }

    if (data[2] != 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchRetrieve: reserved bits non-zero in data[2]");
    }

    if (data[6] || data[7]) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchRetrieve: reserved bits non-zero in data[6-7]");
    }

    status = VerifyRflagConsistency(data, "BatchRetrieve");
    if (!status.ok()) { return status; }

    if (data[9] != (static_cast<std::uint32_t>(DptrType::Batch) << 24)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchRetrieve: data[9] DptrType or reserved bits mismatch");
    }

    if (data[10] >> 16) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchRetrieve: reserved bits non-zero in data[10] bit16-31");
    }
    std::uint16_t batch_number = data[10] & 0xFFFF;
    if (batch_number == 0 || batch_number > kMaxBatchNumber) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "BatchRetrieve: batch_number(" +
                                                               std::to_string(batch_number) +
                                                               ") out of range [1, 110]");
    }

    if (data[8] != static_cast<std::uint32_t>(batch_number) * kBatchEntrySizeBytes) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchRetrieve: data[8](" + std::to_string(data[8]) +
                                 ") != batch_number * " + std::to_string(kBatchEntrySizeBytes));
    }

    std::size_t expected_length =
        (kSqeDwordCount + batch_number * kBatchEntryDwordCount) * sizeof(std::uint32_t);
    if (length != expected_length) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchRetrieve: length(" + std::to_string(length) + ") != expected(" +
                                 std::to_string(expected_length) + ")");
    }

    if ((data[11] & 0x7FFFFFFFU) || data[12] || data[13] || data[14] || data[15]) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "BatchRetrieve: reserved bits non-zero in data[11-15]");
    }

    for (std::uint16_t i = 0; i < batch_number; ++i) {
        const std::uint32_t* base = data + kSqeDwordCount + i * kBatchEntryDwordCount;
        auto entry_status = VerifyBatchEntry(base, "BatchRetrieve", i);
        if (!entry_status.ok()) { return entry_status; }
    }

    return Status::OK();
}

std::size_t KvDeleteProtocol::PackedSize(const SqeRequest& req) const
{
    auto& r = static_cast<const KvDeleteRequest&>(req);
    return (kSqeDwordCount + r.batch_number * kKeyEntryDwordCount) * sizeof(std::uint32_t);
}

Status KvDeleteProtocol::PackSqe(const SqeRequest& req, std::uint32_t* target)
{
    auto& r = static_cast<const KvDeleteRequest&>(req);
    auto status = ValidateRequest(r);
    if (!status.ok()) { return status; }

    target[0] = (r.cid << 16) | (kFixedBits << 14) | static_cast<std::uint32_t>(KvOpcode::Delete);
    if (r.rflag) { target[0] |= (1U << 13); }
    target[1] = r.kv_ns_id;
    target[3] = r.response_buffer_addr & 0xFFFFFFFFULL;
    target[4] = (r.response_buffer_addr >> 32) & 0xFFFFFFFFULL;
    target[5] = r.response_mr_key;
    target[8] = r.batch_number * kKeyEntrySizeBytes;
    target[9] = static_cast<std::uint32_t>(DptrType::Batch) << 24;
    target[10] = r.batch_number;

    for (std::size_t i = 0; i < r.batch_number; ++i) {
        PackEntry(r.keys[i], target + kSqeDwordCount + i * kKeyEntryDwordCount);
    }
    return Status::OK();
}

Status KvDeleteProtocol::ValidateRequest(const KvDeleteRequest& r) const
{
    if (r.batch_number == 0 || r.batch_number > kMaxDeleteBatchNumber) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "batch_number(" + std::to_string(r.batch_number) +
                                 ") must be in range [1, 254] in Delete request");
    }
    if (r.batch_number != r.keys.size()) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "batch_number(" + std::to_string(r.batch_number) +
                                 ") must equal keys.size()(" + std::to_string(r.keys.size()) +
                                 ") in Delete request");
    }
    if (r.rflag) {
        if (r.response_buffer_addr == 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "response_buffer_addr is zero in Delete request");
        }
        if (r.response_mr_key == 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "response_mr_key is zero in Delete request");
        }
    } else {
        if (r.response_buffer_addr != 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "response_buffer_addr must be zero when rflag is false in Delete request");
        }
        if (r.response_mr_key != 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "response_mr_key must be zero when rflag is false in Delete request");
        }
    }
    for (std::size_t i = 0; i < r.batch_number; ++i) {
        if (IsCacheKeyAllZero(r.keys[i])) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "key[" + std::to_string(i) + "] is all zeros in Delete request");
        }
    }
    return Status::OK();
}

void KvDeleteProtocol::PackEntry(const CacheKey& key, std::uint32_t* base)
{
    std::memcpy(&base[0], key.data(), kCacheKeySizeBytes);
}

Status KvDeleteProtocol::UnpackCqe(const std::uint32_t* data, std::uint16_t batch_number,
                                   KvResponse& out) const
{
    UnpackCqeBase(data, out);
    UnpackResultBuffer1Bit(data + kCqeDwordCount, batch_number, out.result_buffer);
    return Status::OK();
}

Status KvDeleteProtocol::VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const
{
    auto status = VerifyFixedBits(data, "Delete");
    if (!status.ok()) { return status; }
    if ((data[0] >> 8) & 0x1F) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Delete: reserved bits non-zero in data[0] bit8-12");
    }

    if (data[2] != 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Delete: reserved bits non-zero in data[2]");
    }

    if (data[6] || data[7]) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Delete: reserved bits non-zero in data[6-7]");
    }

    status = VerifyRflagConsistency(data, "Delete");
    if (!status.ok()) { return status; }

    if (data[9] != (static_cast<std::uint32_t>(DptrType::Batch) << 24)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Delete: data[9] DptrType or reserved bits mismatch");
    }

    if (data[10] >> 16) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Delete: reserved bits non-zero in data[10] bit16-31");
    }
    std::uint16_t batch_number = data[10] & 0xFFFF;
    if (batch_number == 0 || batch_number > kMaxDeleteBatchNumber) {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "Delete: batch_number(" + std::to_string(batch_number) + ") out of range [1, 254]");
    }

    if (data[8] != static_cast<std::uint32_t>(batch_number) * kKeyEntrySizeBytes) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Delete: data[8](" + std::to_string(data[8]) + ") != batch_number * " +
                                 std::to_string(kKeyEntrySizeBytes));
    }

    std::size_t expected_length =
        (kSqeDwordCount + batch_number * kKeyEntryDwordCount) * sizeof(std::uint32_t);
    if (length != expected_length) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Delete: length(" + std::to_string(length) + ") != expected(" +
                                 std::to_string(expected_length) + ")");
    }

    if (data[11] || data[12] || data[13] || data[14] || data[15]) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Delete: reserved bits non-zero in data[11-15]");
    }

    for (std::uint16_t i = 0; i < batch_number; ++i) {
        const std::uint32_t* base = data + kSqeDwordCount + i * kKeyEntryDwordCount;
        auto key_status = VerifyCacheKeyEntry(base, "Delete: entry[" + std::to_string(i) + "]");
        if (!key_status.ok()) { return key_status; }
    }

    return Status::OK();
}

std::size_t KvExistProtocol::PackedSize(const SqeRequest& req) const
{
    auto& r = static_cast<const KvExistRequest&>(req);
    return (kSqeDwordCount + r.batch_number * kKeyEntryDwordCount) * sizeof(std::uint32_t);
}

Status KvExistProtocol::PackSqe(const SqeRequest& req, std::uint32_t* target)
{
    auto& r = static_cast<const KvExistRequest&>(req);
    auto status = ValidateRequest(r);
    if (!status.ok()) { return status; }

    target[0] = (r.cid << 16) | (kFixedBits << 14) | static_cast<std::uint32_t>(KvOpcode::Exist);
    if (r.rflag) { target[0] |= (1U << 13); }
    target[1] = r.kv_ns_id;
    target[3] = r.response_buffer_addr & 0xFFFFFFFFULL;
    target[4] = (r.response_buffer_addr >> 32) & 0xFFFFFFFFULL;
    target[5] = r.response_mr_key;
    target[8] = r.batch_number * kKeyEntrySizeBytes;
    target[9] = static_cast<std::uint32_t>(DptrType::Batch) << 24;
    target[10] = r.batch_number;
    if (r.sc) { target[10] |= (1U << 16); }

    for (std::size_t i = 0; i < r.batch_number; ++i) {
        PackEntry(r.keys[i], target + kSqeDwordCount + i * kKeyEntryDwordCount);
    }
    return Status::OK();
}

Status KvExistProtocol::ValidateRequest(const KvExistRequest& r) const
{
    if (r.batch_number == 0 || r.batch_number > kMaxExistBatchNumber) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "batch_number(" + std::to_string(r.batch_number) +
                                 ") must be in range [1, 256] in Exist request");
    }
    if (r.batch_number != r.keys.size()) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "batch_number(" + std::to_string(r.batch_number) +
                                 ") must equal keys.size()(" + std::to_string(r.keys.size()) +
                                 ") in Exist request");
    }
    if (r.rflag) {
        if (r.response_buffer_addr == 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "response_buffer_addr is zero in Exist request");
        }
        if (r.response_mr_key == 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "response_mr_key is zero in Exist request");
        }
    } else {
        if (r.response_buffer_addr != 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "response_buffer_addr must be zero when rflag is false in Exist request");
        }
        if (r.response_mr_key != 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "response_mr_key must be zero when rflag is false in Exist request");
        }
    }
    for (std::size_t i = 0; i < r.batch_number; ++i) {
        if (IsCacheKeyAllZero(r.keys[i])) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "key[" + std::to_string(i) + "] is all zeros in Exist request");
        }
    }
    return Status::OK();
}

void KvExistProtocol::PackEntry(const CacheKey& key, std::uint32_t* base)
{
    std::memcpy(&base[0], key.data(), kCacheKeySizeBytes);
}

Status KvExistProtocol::UnpackCqe(const std::uint32_t* data, std::uint16_t batch_number,
                                  KvResponse& out) const
{
    UnpackCqeBase(data, out);
    out.existing_key_number = data[0] & 0xFFFF;
    UnpackResultBuffer1Bit(data + kCqeDwordCount, batch_number, out.result_buffer);
    return Status::OK();
}

Status KvExistProtocol::VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const
{
    auto status = VerifyFixedBits(data, "Exist");
    if (!status.ok()) { return status; }
    if ((data[0] >> 8) & 0x1F) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Exist: reserved bits non-zero in data[0] bit8-12");
    }

    if (data[2] != 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Exist: reserved bits non-zero in data[2]");
    }

    if (data[6] || data[7]) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Exist: reserved bits non-zero in data[6-7]");
    }

    status = VerifyRflagConsistency(data, "Exist");
    if (!status.ok()) { return status; }

    if (data[9] != (static_cast<std::uint32_t>(DptrType::Batch) << 24)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Exist: data[9] DptrType or reserved bits mismatch");
    }

    if (data[10] >> 17) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Exist: reserved bits non-zero in data[10] bit17-31");
    }
    std::uint16_t batch_number = data[10] & 0xFFFF;
    if (batch_number == 0 || batch_number > kMaxExistBatchNumber) {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "Exist: batch_number(" + std::to_string(batch_number) + ") out of range [1, 256]");
    }

    if (data[8] != static_cast<std::uint32_t>(batch_number) * kKeyEntrySizeBytes) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Exist: data[8](" + std::to_string(data[8]) + ") != batch_number * " +
                                 std::to_string(kKeyEntrySizeBytes));
    }

    std::size_t expected_length =
        (kSqeDwordCount + batch_number * kKeyEntryDwordCount) * sizeof(std::uint32_t);
    if (length != expected_length) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Exist: length(" + std::to_string(length) + ") != expected(" +
                                 std::to_string(expected_length) + ")");
    }

    if (data[11] || data[12] || data[13] || data[14] || data[15]) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Exist: reserved bits non-zero in data[11-15]");
    }

    for (std::uint16_t i = 0; i < batch_number; ++i) {
        const std::uint32_t* base = data + kSqeDwordCount + i * kKeyEntryDwordCount;
        auto key_status = VerifyCacheKeyEntry(base, "Exist: entry[" + std::to_string(i) + "]");
        if (!key_status.ok()) { return key_status; }
    }

    return Status::OK();
}

Status KvKeepAliveProtocol::PackSqe(const SqeRequest& req, std::uint32_t* target)
{
    auto& r = static_cast<const KvKeepAliveRequest&>(req);
    auto status = ValidateRequest(r);
    if (!status.ok()) { return status; }

    target[0] = (r.cid << 16) | static_cast<std::uint32_t>(KvOpcode::KeepAlive);
    if (r.rflag) { target[0] |= (1U << 13); }
    target[3] = r.response_buffer_addr & 0xFFFFFFFFULL;
    target[4] = (r.response_buffer_addr >> 32) & 0xFFFFFFFFULL;
    target[5] = r.response_mr_key;
    return Status::OK();
}

Status KvKeepAliveProtocol::ValidateRequest(const KvKeepAliveRequest& r) const
{
    if (r.rflag) {
        if (r.response_buffer_addr == 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "response_buffer_addr is zero in KeepAlive request");
        }
        if (r.response_mr_key == 0) [[unlikely]] {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "response_mr_key is zero in KeepAlive request");
        }
    } else {
        if (r.response_buffer_addr != 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "response_buffer_addr must be zero when rflag is false in KeepAlive request");
        }
        if (r.response_mr_key != 0) [[unlikely]] {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "response_mr_key must be zero when rflag is false in KeepAlive request");
        }
    }
    return Status::OK();
}

Status KvKeepAliveProtocol::VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const
{
    constexpr std::size_t kExpectedLength = kSqeDwordCount * sizeof(std::uint32_t);
    if (length != kExpectedLength) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "KeepAlive: length(" + std::to_string(length) + ") != expected(" +
                                 std::to_string(kExpectedLength) + ")");
    }

    if (data[0] & 0xDF00U) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "KeepAlive: reserved bits non-zero in data[0] bit8-12 or bit14-15");
    }

    if (data[1] || data[2]) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "KeepAlive: reserved bits non-zero in data[1-2]");
    }

    auto status = VerifyRflagConsistency(data, "KeepAlive");
    if (!status.ok()) { return status; }

    for (std::size_t i = 6; i < kSqeDwordCount; ++i) {
        if (data[i] != 0) {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "KeepAlive: reserved bits non-zero in data[" + std::to_string(i) + "]");
        }
    }

    return Status::OK();
}

ProtocolManager::ProtocolManager()
{
    protocols_[KvOpcode::Store] = std::make_unique<KvStoreProtocol>();
    protocols_[KvOpcode::Retrieve] = std::make_unique<KvRetrieveProtocol>();
    protocols_[KvOpcode::BatchStore] = std::make_unique<KvBatchStoreProtocol>();
    protocols_[KvOpcode::BatchRetrieve] = std::make_unique<KvBatchRetrieveProtocol>();
    protocols_[KvOpcode::Delete] = std::make_unique<KvDeleteProtocol>();
    protocols_[KvOpcode::Exist] = std::make_unique<KvExistProtocol>();
    protocols_[KvOpcode::KeepAlive] = std::make_unique<KvKeepAliveProtocol>();
}

std::size_t ProtocolManager::GetPackedSize(KvOpcode opcode, const SqeRequest& req) const
{
    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) { return 0; }
    return proto->PackedSize(req);
}

Status ProtocolManager::PackRequest(void* data_ptr, KvOpcode opcode, const SqeRequest& req)
{
    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "unknown opcode in PackRequest");
    }
    return proto->PackSqe(req, static_cast<std::uint32_t*>(data_ptr));
}

Status ProtocolManager::UnpackResponse(const void* data_ptr, KvOpcode opcode,
                                       std::uint16_t batch_number, KvResponse& out)
{
    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "unknown opcode in UnpackResponse");
    }
    auto* data = static_cast<const std::uint32_t*>(data_ptr);
    return proto->UnpackCqe(data, batch_number, out);
}

Status ProtocolManager::PollResponseCid(const void* data_ptr, std::uint16_t& cid) const
{
    auto* data = static_cast<const std::uint32_t*>(data_ptr);
    cid = __atomic_load_n(data + 3, __ATOMIC_ACQUIRE) & 0xFFFF;
    return Status::OK();
}

Status ProtocolManager::VerifyPackedBuffer(const void* data_ptr, std::size_t length)
{
    if (!data_ptr || length < kSqeDwordCount * sizeof(std::uint32_t)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "VerifyPackedBuffer: invalid data_ptr or length too small");
    }

    auto* data = static_cast<const std::uint32_t*>(data_ptr);
    auto opcode = static_cast<KvOpcode>(data[0] & 0xFF);

    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "VerifyPackedBuffer: unknown opcode " +
                                 std::to_string(static_cast<std::uint8_t>(opcode)));
    }
    return proto->VerifyPackedBuffer(data, length);
}

KvProtocol* ProtocolManager::GetProtocol(KvOpcode opcode) const
{
    auto it = protocols_.find(opcode);
    return it != protocols_.end() ? it->second.get() : nullptr;
}

}  // namespace kv
