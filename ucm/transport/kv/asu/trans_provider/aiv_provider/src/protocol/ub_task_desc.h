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
#include "src/protocol/ub_signal_value.h"

namespace umc::comm {

constexpr uint16_t kTaskDescMagicU16 = 0x5A7A;
constexpr uint8_t kTaskDescVersionV1 = 1;
constexpr uint16_t kMaxTaskBatchEntries = 256;

enum kv_task_op_t : uint8_t {
    TASK_OP_INVALID = 0x00,
    TASK_OP_PUT = 0x01,
    TASK_OP_GET = 0x02,
    TASK_OP_BATCH_GET = 0x06,
    TASK_OP_BATCH_PUT = 0x07,
    TASK_OP_DELETE = 0x0B,
    TASK_OP_EXIST = 0x0F,
};

enum kv_task_status_t : uint16_t {
    TASK_STATUS_DONE = 0,
    TASK_STATUS_FAILED = 1,
    TASK_STATUS_NOT_FOUND = 2,
    TASK_STATUS_TIMEOUT = 3,
};

constexpr uint32_t TASK_FLAG_NONE = 0x0;
constexpr uint32_t TASK_FLAG_NEED_NOTIFY = 0x1;

#pragma pack(push, 1)
struct TaskDesc {
    uint16_t magic;               // [  0,  2)  == kTaskDescMagicU16
    uint8_t version;              // [  2,  3)  == kTaskDescVersionV1
    uint8_t op;                   // [  3,  4)  kv_task_op_t
    uint32_t client_id;           // [  4,  8)
    uint64_t task_id;             // [  8, 16)  completion correlation id
    uint64_t block_id;            // [ 16, 24)  KV block key
    uint64_t storage_offset;      // [ 24, 32)
    uint32_t kvnsid;              // [ 32, 36)
    uint32_t length;              // [ 36, 40)
    uint32_t flags;               // [ 40, 44)  TASK_FLAG_*
    uint32_t client_data_mr_id;   // [ 44, 48)
    uint64_t client_data_addr;    // [ 48, 56)
    uint32_t client_data_rkey;    // [ 56, 60)
    uint32_t client_flag_rkey;    // [ 60, 64)  client flag MR token/rkey
    uint64_t client_flag_addr;    // [ 64, 72)
    uint32_t client_jetty_uasid;  // [ 72, 76)
    uint32_t client_jetty_id;     // [ 76, 80)  client jetty id
    uint64_t user_cookie;         // [ 80, 88)
    uint32_t checksum;            // [ 88, 92)
    uint16_t batch_index;         // [ 92, 94)
    uint16_t batch_total;         // [ 94, 96)
    uint8_t reserved[32];         // [ 96,128)
};
#pragma pack(pop)
static_assert(sizeof(TaskDesc) == 128, "TaskDesc must be 128 bytes");
static_assert(offsetof(TaskDesc, magic) == 0, "");
static_assert(offsetof(TaskDesc, version) == 2, "");
static_assert(offsetof(TaskDesc, op) == 3, "");
static_assert(offsetof(TaskDesc, client_id) == 4, "");
static_assert(offsetof(TaskDesc, task_id) == 8, "");
static_assert(offsetof(TaskDesc, block_id) == 16, "");
static_assert(offsetof(TaskDesc, storage_offset) == 24, "");
static_assert(offsetof(TaskDesc, kvnsid) == 32, "");
static_assert(offsetof(TaskDesc, length) == 36, "");
static_assert(offsetof(TaskDesc, flags) == 40, "");
static_assert(offsetof(TaskDesc, client_data_mr_id) == 44, "");
static_assert(offsetof(TaskDesc, client_data_addr) == 48, "");
static_assert(offsetof(TaskDesc, client_data_rkey) == 56, "");
static_assert(offsetof(TaskDesc, client_flag_rkey) == 60, "");
static_assert(offsetof(TaskDesc, client_flag_addr) == 64, "");
static_assert(offsetof(TaskDesc, client_jetty_uasid) == 72, "");
static_assert(offsetof(TaskDesc, client_jetty_id) == 76, "");
static_assert(offsetof(TaskDesc, user_cookie) == 80, "");
static_assert(offsetof(TaskDesc, checksum) == 88, "");
static_assert(offsetof(TaskDesc, batch_index) == 92, "");
static_assert(offsetof(TaskDesc, batch_total) == 94, "");
static_assert(offsetof(TaskDesc, reserved) == 96, "");

#pragma pack(push, 1)
struct TaskBatchEntry {
    uint64_t block_id;           // [ 0, 8)  KV block key
    uint64_t storage_offset;     // [ 8,16)
    uint64_t client_data_addr;   // [16,24)
    uint32_t length;             // [24,28)
    uint32_t client_data_mr_id;  // [28,32)
    uint32_t client_data_rkey;   // [32,36)
    uint32_t reserved;           // [36,40)
};
#pragma pack(pop)
static_assert(sizeof(TaskBatchEntry) == 40, "TaskBatchEntry must be 40 bytes");
static_assert(offsetof(TaskBatchEntry, block_id) == 0, "");
static_assert(offsetof(TaskBatchEntry, storage_offset) == 8, "");
static_assert(offsetof(TaskBatchEntry, client_data_addr) == 16, "");
static_assert(offsetof(TaskBatchEntry, length) == 24, "");
static_assert(offsetof(TaskBatchEntry, client_data_mr_id) == 28, "");
static_assert(offsetof(TaskBatchEntry, client_data_rkey) == 32, "");
static_assert(offsetof(TaskBatchEntry, reserved) == 36, "");

constexpr size_t kMaxTaskBatchPayloadBytes =
    sizeof(TaskDesc) + static_cast<size_t>(kMaxTaskBatchEntries) * sizeof(TaskBatchEntry);

#pragma pack(push, 1)
struct CompletionMsg {
    uint16_t magic;        // [ 0, 2)
    uint8_t version;       // [ 2, 3)  == kUbSignalVersionV1
    uint8_t _pad0;         // [ 3, 4)
    uint16_t status;       // [ 4, 6)  kv_task_status_t
    uint16_t _pad1;        // [ 6, 8)
    uint64_t task_id;      // [ 8,16)
    uint32_t bytes;        // [16,20)
    uint32_t error_code;   // [20,24)
    uint64_t user_cookie;  // [24,32)
};
#pragma pack(pop)
static_assert(sizeof(CompletionMsg) == 32, "CompletionMsg must be 32 bytes");
static_assert(offsetof(CompletionMsg, magic) == 0, "");
static_assert(offsetof(CompletionMsg, version) == 2, "");
static_assert(offsetof(CompletionMsg, status) == 4, "");
static_assert(offsetof(CompletionMsg, task_id) == 8, "");
static_assert(offsetof(CompletionMsg, bytes) == 16, "");
static_assert(offsetof(CompletionMsg, error_code) == 20, "");
static_assert(offsetof(CompletionMsg, user_cookie) == 24, "");

inline uint64_t EncodeCompletionHead(uint16_t status)
{
    CompletionMsg c{};
    c.magic = kUbSignalMagicU16;
    c.version = kUbSignalVersionV1;
    c.status = status;
    uint64_t raw = 0;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&c);
    for (int i = 0; i < 8; ++i) { raw |= static_cast<uint64_t>(p[i]) << (i * 8); }
    return raw;
}

inline uint32_t ComputeTaskDescChecksum(const TaskDesc& t)
{
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&t);
    uint32_t sum = 0;
    for (size_t i = 0; i < offsetof(TaskDesc, checksum); ++i) {
        sum += static_cast<uint32_t>(p[i]) * (static_cast<uint32_t>(i) + 1u);
    }
    return sum == 0 ? 1u : sum;
}

inline bool IsValidTaskDesc(const TaskDesc& t)
{
    if (t.magic != kTaskDescMagicU16 || t.version != kTaskDescVersionV1) return false;
    if (t.checksum != 0 && t.checksum != ComputeTaskDescChecksum(t)) return false;
    return true;
}

}  // namespace umc::comm
