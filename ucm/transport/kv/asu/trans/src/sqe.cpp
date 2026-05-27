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
#include "sqe.h"
#include <algorithm>
#include <cstring>

namespace UC::ASU {

Status KvStoreSqe::Pack(const SqeRequest& req)
{
    auto& r = static_cast<const KvStoreRequest&>(req);
    dwords.assign(kSqeDwordCount, 0);

    // Dword 0: CID[31:16] | Fixed[15:14]=0b11 | Reserved[13:8] | Opcode[7:0]=0x01
    dwords[0] = (r.cid << 16) | (kFixedBits << 14) | static_cast<std::uint32_t>(SqeOpcode::Store);

    // Dword 1: kv_ns_id[31:0]
    dwords[1] = r.kv_ns_id;

    // Dword 2: DTYPE[15:13] | DSPEC[12:8] | Reserved[7:0]
    dwords[2] = ((r.dtype & 0x7) << 13) | ((r.dspec & 0x1F) << 8);

    // Dwords 3-5: reserved (zero)

    // Dword 6-7: DPTR.buffer[63:0] - data buffer address
    dwords[6] = r.buffer_addr & 0xFFFFFFFFULL;
    dwords[7] = (r.buffer_addr >> 32) & 0xFFFFFFFFULL;

    // Dword 8: key[0][31:24] = MR_KEY low 8 bits | length[23:0] = buffer length
    dwords[8] = ((r.mr_key & 0xFF) << 24) | (r.buffer_length & 0xFFFFFF);

    // Dword 9: Type[31:24] = 0x40 | key[3:1][23:0] = MR_KEY high 24 bits
    dwords[9] =
        (static_cast<std::uint32_t>(DptrType::Standard) << 24) | ((r.mr_key >> 8) & 0xFFFFFF);

    // Dword 10: offset[31:0]
    dwords[10] = r.offset;

    // Dword 11: LR[31] | Reserved[30:24] | Length[23:0]
    dwords[11] = (r.lr ? (1U << 31) : 0) | (r.length & 0xFFFFFF);

    // Dwords 12-15: key[15:0] - 16 bytes key, low-byte aligned
    std::size_t key_len = std::min(r.key.size(), static_cast<std::size_t>(16));
    if (key_len > 0) { std::memcpy(&dwords[12], r.key.data(), key_len); }
    return Status::OK();
}

Status KvStoreSqe::Validate() const
{
    if (dwords.size() < kSqeDwordCount) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "SQE not packed");
    }
    if (dwords[6] == 0 && dwords[7] == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "buffer_addr is zero");
    }
    std::uint32_t buffer_length = dwords[8] & 0xFFFFFF;
    if (buffer_length == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "buffer_length is zero");
    }
    if (buffer_length % kAlignmentBytes != 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "buffer_length must be 512B aligned");
    }
    if (dwords[10] % kAlignmentBytes != 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "offset must be 512B aligned");
    }
    std::uint32_t length = dwords[11] & 0xFFFFFF;
    if (length == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "length is 1-based, must be non-zero");
    }
    bool key_empty = true;
    for (int i = 0; i < 4; ++i) {
        if (dwords[12 + i] != 0) {
            key_empty = false;
            break;
        }
    }
    if (key_empty) { return Status::Error(StatusCode::INVALID_ARGUMENT, "key is empty"); }
    return Status::OK();
}

Status KvRetrieveSqe::Pack(const SqeRequest& req)
{
    auto& r = static_cast<const KvRetrieveRequest&>(req);
    dwords.assign(kSqeDwordCount, 0);

    // Dword 0: CID[31:16] | Fixed[15:14]=0b11 | Reserved[13:8] | Opcode[7:0]=0x02
    dwords[0] =
        (r.cid << 16) | (kFixedBits << 14) | static_cast<std::uint32_t>(SqeOpcode::Retrieve);

    // Dword 1: kv_ns_id[31:0]
    dwords[1] = r.kv_ns_id;

    // Dword 2: Reserved[15:0]
    // Dwords 3-5: reserved (zero)

    // Dword 6-7: DPTR.buffer[63:0] - data buffer address
    dwords[6] = r.buffer_addr & 0xFFFFFFFFULL;
    dwords[7] = (r.buffer_addr >> 32) & 0xFFFFFFFFULL;

    // Dword 8: key[0][31:24] = MR_KEY low 8 bits | length[23:0] = buffer length
    dwords[8] = ((r.mr_key & 0xFF) << 24) | (r.buffer_length & 0xFFFFFF);

    // Dword 9: Type[31:24] = 0x40 | key[3:1][23:0] = MR_KEY high 24 bits
    dwords[9] =
        (static_cast<std::uint32_t>(DptrType::Standard) << 24) | ((r.mr_key >> 8) & 0xFFFFFF);

    // Dword 10: offset[31:0]
    dwords[10] = r.offset;

    // Dword 11: LR[31] | Reserved[30:24] | Length[23:0]
    dwords[11] = (r.lr ? (1U << 31) : 0) | (r.length & 0xFFFFFF);

    // Dwords 12-15: key[15:0] - 16 bytes key, low-byte aligned
    std::size_t key_len = std::min(r.key.size(), static_cast<std::size_t>(16));
    if (key_len > 0) { std::memcpy(&dwords[12], r.key.data(), key_len); }
    return Status::OK();
}

Status KvRetrieveSqe::Validate() const
{
    if (dwords.size() < kSqeDwordCount) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "SQE not packed");
    }
    if (dwords[6] == 0 && dwords[7] == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "buffer_addr is zero");
    }
    std::uint32_t buffer_length = dwords[8] & 0xFFFFFF;
    if (buffer_length == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "buffer_length is zero");
    }
    if (buffer_length % kAlignmentBytes != 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "buffer_length must be 512B aligned");
    }
    if (dwords[10] % kAlignmentBytes != 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "offset must be 512B aligned");
    }
    std::uint32_t length = dwords[11] & 0xFFFFFF;
    if (length == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "length is 1-based, must be non-zero");
    }
    bool key_empty = true;
    for (int i = 0; i < 4; ++i) {
        if (dwords[12 + i] != 0) {
            key_empty = false;
            break;
        }
    }
    if (key_empty) { return Status::Error(StatusCode::INVALID_ARGUMENT, "key is empty"); }
    return Status::OK();
}

Status KvBatchStoreSqe::Pack(const SqeRequest& req)
{
    auto& r = static_cast<const KvBatchStoreRequest&>(req);
    if (r.batch_number > r.entries.size()) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "batch_number exceeds entries.size()");
    }
    dwords.assign(kSqeDwordCount + r.batch_number * kBatchEntryDwordCount, 0);

    // Dword 0: CID[31:16] | Fixed[15:14]=0b11 | Rflag[13] | Reserved[12:8] | Opcode[7:0]=0x45
    dwords[0] =
        (r.cid << 16) | (kFixedBits << 14) | static_cast<std::uint32_t>(SqeOpcode::BatchStore);
    if (r.rflag) { dwords[0] |= (1U << 13); }

    // Dword 1: kv_ns_id[31:0]
    dwords[1] = r.kv_ns_id;

    // Dword 2: DTYPE[15:13] | DSPEC[12:8]
    dwords[2] = ((r.dtype & 0x7) << 13) | ((r.dspec & 0x1F) << 8);

    // Dword 3-4: Response Buffer Address[63:0]
    dwords[3] = r.response_buffer_addr & 0xFFFFFFFFULL;
    dwords[4] = (r.response_buffer_addr >> 32) & 0xFFFFFFFFULL;

    // Dword 5: Response Buffer MR_Key[31:0]
    dwords[5] = r.response_mr_key;

    // Dword 6-7: DPTR.buffer = 0 (fixed)

    // Dword 8: DPTR.length = Batch Number * 36
    dwords[8] = r.batch_number * kBatchEntrySizeBytes;

    // Dword 9: DPTR.Type = 0x1
    dwords[9] = static_cast<std::uint32_t>(DptrType::Batch) << 24;

    // Dword 10: Batch Number
    dwords[10] = r.batch_number;

    // Dword 11: LR[31]
    if (r.lr) { dwords[11] |= (1U << 31); }

    // Dwords 12-15: reserved (zero)

    // Pack batch entries
    for (std::size_t i = 0; i < r.batch_number; ++i) { PackEntry(r.entries[i], i); }
    return Status::OK();
}

Status KvBatchStoreSqe::Validate() const
{
    if (dwords.size() < kSqeDwordCount) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "SQE not packed");
    }
    if (dwords[3] == 0 && dwords[4] == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "response_buffer_addr is zero");
    }
    std::uint32_t batch_number = dwords[10] & 0xFFFF;
    if (batch_number == 0 || batch_number > kMaxBatchNumber) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "batch_number must be in range [1, 227]");
    }
    std::uint32_t dptr_length = dwords[8] & 0xFFFFFF;
    if (dptr_length != batch_number * kBatchEntrySizeBytes) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "DPTR.length must equal batch_number * 36");
    }
    std::size_t expected_size = kSqeDwordCount + batch_number * kBatchEntryDwordCount;
    if (dwords.size() != expected_size) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "missing batch entries");
    }
    return Status::OK();
}

void KvBatchStoreSqe::PackEntry(const KvBatchStoreEntry& entry, std::size_t index)
{
    std::size_t base = kSqeDwordCount + index * kBatchEntryDwordCount;

    // Entry Dword 0: offset
    dwords[base + 0] = entry.offset;

    // Entry Dword 1: key[15:0]
    std::size_t key_len = std::min(entry.key.size(), static_cast<std::size_t>(16));
    if (key_len > 0) { std::memcpy(&dwords[base + 1], entry.key.data(), key_len); }

    // Entry Dword 5-6: Buffer Address[63:0]
    dwords[base + 5] = entry.buffer_addr & 0xFFFFFFFFULL;
    dwords[base + 6] = (entry.buffer_addr >> 32) & 0xFFFFFFFFULL;

    // Entry Dword 7: MR_KEY[0][31:24] = MR_KEY low 8 bits | Length[23:0]
    dwords[base + 7] = ((entry.mr_key & 0xFF) << 24) | (entry.length & 0xFFFFFF);

    // Entry Dword 8: DPTR.Type = 0x40 | MR_KEY[3:1][23:0] = MR_KEY high 24 bits
    dwords[base + 8] =
        (static_cast<std::uint32_t>(DptrType::Standard) << 24) | ((entry.mr_key >> 8) & 0xFFFFFF);
}

Status KvBatchRetrieveSqe::Pack(const SqeRequest& req)
{
    auto& r = static_cast<const KvBatchRetrieveRequest&>(req);
    if (r.batch_number > r.entries.size()) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "batch_number exceeds entries.size()");
    }
    dwords.assign(kSqeDwordCount + r.batch_number * kBatchEntryDwordCount, 0);

    // Dword 0: CID[31:16] | Fixed[15:14]=0b11 | Rflag[13] | Reserved[12:8] | Opcode[7:0]=0x46
    dwords[0] =
        (r.cid << 16) | (kFixedBits << 14) | static_cast<std::uint32_t>(SqeOpcode::BatchRetrieve);
    if (r.rflag) { dwords[0] |= (1U << 13); }

    // Dword 1: kv_ns_id[31:0]
    dwords[1] = r.kv_ns_id;

    // Dword 2: Reserved[15:0]
    // Dwords 3-4: Response Buffer Address[63:0]
    dwords[3] = r.response_buffer_addr & 0xFFFFFFFFULL;
    dwords[4] = (r.response_buffer_addr >> 32) & 0xFFFFFFFFULL;

    // Dword 5: Response Buffer MR_Key[31:0]
    dwords[5] = r.response_mr_key;

    // Dword 6-7: DPTR.buffer = 0 (fixed)

    // Dword 8: DPTR.length = Batch Number * 36
    dwords[8] = r.batch_number * kBatchEntrySizeBytes;

    // Dword 9: DPTR.Type = 0x1
    dwords[9] = static_cast<std::uint32_t>(DptrType::Batch) << 24;

    // Dword 10: Batch Number
    dwords[10] = r.batch_number;

    // Dword 11: LR[31]
    if (r.lr) { dwords[11] |= (1U << 31); }

    // Dwords 12-15: reserved (zero)

    // Pack batch entries
    for (std::size_t i = 0; i < r.batch_number; ++i) { PackEntry(r.entries[i], i); }
    return Status::OK();
}

Status KvBatchRetrieveSqe::Validate() const
{
    if (dwords.size() < kSqeDwordCount) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "SQE not packed");
    }
    if (dwords[3] == 0 && dwords[4] == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "response_buffer_addr is zero");
    }
    std::uint32_t batch_number = dwords[10] & 0xFFFF;
    if (batch_number == 0 || batch_number > kMaxBatchNumber) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "batch_number must be in range [1, 227]");
    }
    std::uint32_t dptr_length = dwords[8] & 0xFFFFFF;
    if (dptr_length != batch_number * kBatchEntrySizeBytes) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "DPTR.length must equal batch_number * 36");
    }
    std::size_t expected_size = kSqeDwordCount + batch_number * kBatchEntryDwordCount;
    if (dwords.size() != expected_size) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "missing batch entries");
    }
    return Status::OK();
}

void KvBatchRetrieveSqe::PackEntry(const KvBatchRetrieveEntry& entry, std::size_t index)
{
    std::size_t base = kSqeDwordCount + index * kBatchEntryDwordCount;

    // Entry Dword 0: offset
    dwords[base + 0] = entry.offset;

    // Entry Dword 1: key[15:0]
    std::size_t key_len = std::min(entry.key.size(), static_cast<std::size_t>(16));
    if (key_len > 0) { std::memcpy(&dwords[base + 1], entry.key.data(), key_len); }

    // Entry Dword 5-6: Buffer Address[63:0]
    dwords[base + 5] = entry.buffer_addr & 0xFFFFFFFFULL;
    dwords[base + 6] = (entry.buffer_addr >> 32) & 0xFFFFFFFFULL;

    // Entry Dword 7: MR_KEY[0][31:24] = MR_KEY low 8 bits | Length[23:0]
    dwords[base + 7] = ((entry.mr_key & 0xFF) << 24) | (entry.length & 0xFFFFFF);

    // Entry Dword 8: DPTR.Type = 0x40 | MR_KEY[3:1][23:0] = MR_KEY high 24 bits
    dwords[base + 8] =
        (static_cast<std::uint32_t>(DptrType::Standard) << 24) | ((entry.mr_key >> 8) & 0xFFFFFF);
}

Status KvDeleteSqe::Pack(const SqeRequest& req)
{
    auto& r = static_cast<const KvDeleteRequest&>(req);
    if (r.batch_number > r.keys.size()) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "batch_number exceeds keys.size()");
    }
    dwords.assign(kSqeDwordCount + r.batch_number * kDeleteEntryDwordCount, 0);

    // Dword 0: CID[31:16] | Fixed[15:14]=0b11 | Rflag[13] | Reserved[12:8] | Opcode[7:0]=0x08
    dwords[0] = (r.cid << 16) | (kFixedBits << 14) | static_cast<std::uint32_t>(SqeOpcode::Delete);
    if (r.rflag) { dwords[0] |= (1U << 13); }

    // Dword 1: kv_ns_id[31:0]
    dwords[1] = r.kv_ns_id;

    // Dword 2: Reserved[15:0]
    // Dwords 3-4: Response Buffer Address[63:0]
    dwords[3] = r.response_buffer_addr & 0xFFFFFFFFULL;
    dwords[4] = (r.response_buffer_addr >> 32) & 0xFFFFFFFFULL;

    // Dword 5: Response Buffer MR_Key[31:0]
    dwords[5] = r.response_mr_key;

    // Dword 6-7: DPTR.buffer = 0 (fixed)

    // Dword 8: DPTR.length = Batch Number * 16
    dwords[8] = r.batch_number * kDeleteEntrySizeBytes;

    // Dword 9: DPTR.Type = 0x1
    dwords[9] = static_cast<std::uint32_t>(DptrType::Batch) << 24;

    // Dword 10: Batch Number
    dwords[10] = r.batch_number;

    // Dwords 11-15: reserved (zero)

    // Pack delete entries
    for (std::size_t i = 0; i < r.batch_number; ++i) { PackEntry(r.keys[i], i); }
    return Status::OK();
}

Status KvDeleteSqe::Validate() const
{
    if (dwords.size() < kSqeDwordCount) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "SQE not packed");
    }
    if (dwords[3] == 0 && dwords[4] == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "response_buffer_addr is zero");
    }
    std::uint32_t batch_number = dwords[10] & 0xFFFF;
    if (batch_number == 0 || batch_number > kMaxBatchNumber) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "batch_number must be in range [1, 227]");
    }
    std::uint32_t dptr_length = dwords[8] & 0xFFFFFF;
    if (dptr_length != batch_number * kDeleteEntrySizeBytes) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "DPTR.length must equal batch_number * 16");
    }
    std::size_t expected_size = kSqeDwordCount + batch_number * kDeleteEntryDwordCount;
    if (dwords.size() != expected_size) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "missing delete entries");
    }
    return Status::OK();
}

void KvDeleteSqe::PackEntry(const std::string& key, std::size_t index)
{
    std::size_t base = kSqeDwordCount + index * kDeleteEntryDwordCount;

    std::size_t key_len = std::min(key.size(), static_cast<std::size_t>(16));
    if (key_len > 0) { std::memcpy(&dwords[base], key.data(), key_len); }
}

Status KvExistSqe::Pack(const SqeRequest& req)
{
    auto& r = static_cast<const KvExistRequest&>(req);
    if (r.batch_number > r.keys.size()) [[unlikely]] {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "batch_number exceeds keys.size()");
    }
    dwords.assign(kSqeDwordCount + r.batch_number * kDeleteEntryDwordCount, 0);

    // Dword 0: CID[31:16] | Fixed[15:14]=0b11 | Rflag[13] | Reserved[12:8] | Opcode[7:0]=0x0C
    dwords[0] = (r.cid << 16) | (kFixedBits << 14) | static_cast<std::uint32_t>(SqeOpcode::Exist);
    if (r.rflag) { dwords[0] |= (1U << 13); }

    // Dword 1: kv_ns_id[31:0]
    dwords[1] = r.kv_ns_id;

    // Dword 2: Reserved[15:0]
    // Dwords 3-4: Response Buffer Address[63:0]
    dwords[3] = r.response_buffer_addr & 0xFFFFFFFFULL;
    dwords[4] = (r.response_buffer_addr >> 32) & 0xFFFFFFFFULL;

    // Dword 5: Response Buffer MR_Key[31:0]
    dwords[5] = r.response_mr_key;

    // Dword 6-7: DPTR.buffer = 0 (fixed)

    // Dword 8: DPTR.length = Batch Number * 16
    dwords[8] = r.batch_number * kDeleteEntrySizeBytes;

    // Dword 9: DPTR.Type = 0x1
    dwords[9] = static_cast<std::uint32_t>(DptrType::Batch) << 24;

    // Dword 10: SC[16] | Batch Number[15:0]
    dwords[10] = r.batch_number;
    if (r.sc) { dwords[10] |= (1U << 16); }

    // Dwords 11-15: reserved (zero)

    // Pack exist entries
    for (std::size_t i = 0; i < r.batch_number; ++i) { PackEntry(r.keys[i], i); }
    return Status::OK();
}

Status KvExistSqe::Validate() const
{
    if (dwords.size() < kSqeDwordCount) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "SQE not packed");
    }
    if (dwords[3] == 0 && dwords[4] == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "response_buffer_addr is zero");
    }
    std::uint32_t batch_number = dwords[10] & 0xFFFF;
    if (batch_number == 0 || batch_number > kMaxBatchNumber) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "batch_number must be in range [1, 227]");
    }
    std::uint32_t dptr_length = dwords[8] & 0xFFFFFF;
    if (dptr_length != batch_number * kDeleteEntrySizeBytes) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "DPTR.length must equal batch_number * 16");
    }
    std::size_t expected_size = kSqeDwordCount + batch_number * kDeleteEntryDwordCount;
    if (dwords.size() != expected_size) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "missing exist entries");
    }
    return Status::OK();
}

void KvExistSqe::PackEntry(const std::string& key, std::size_t index)
{
    std::size_t base = kSqeDwordCount + index * kDeleteEntryDwordCount;

    std::size_t key_len = std::min(key.size(), static_cast<std::size_t>(16));
    if (key_len > 0) { std::memcpy(&dwords[base], key.data(), key_len); }
}

Status KvKeepAliveSqe::Pack(const SqeRequest& req)
{
    auto& r = static_cast<const KvKeepAliveRequest&>(req);
    dwords.assign(kSqeDwordCount, 0);

    // Dword 0: CID[31:16] | Rflag[13] | Opcode[7:0]=0xF4
    dwords[0] = (r.cid << 16) | static_cast<std::uint32_t>(SqeOpcode::KeepAlive);
    if (r.rflag) { dwords[0] |= (1U << 13); }

    // Dword 1-2: Reserved (zero)
    // Dwords 3-4: Response Buffer Address[63:0]
    dwords[3] = r.response_buffer_addr & 0xFFFFFFFFULL;
    dwords[4] = (r.response_buffer_addr >> 32) & 0xFFFFFFFFULL;

    // Dword 5: Response Buffer MR_Key[31:0]
    dwords[5] = r.response_mr_key;

    // Dwords 6-15: reserved (zero)
    return Status::OK();
}

Status KvKeepAliveSqe::Validate() const
{
    if (dwords.size() < kSqeDwordCount) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "SQE not packed");
    }
    bool rflag = (dwords[0] >> 13) & 0x1;
    if (rflag && dwords[3] == 0 && dwords[4] == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "response_buffer_addr is zero");
    }
    return Status::OK();
}

}  // namespace UC::ASU
