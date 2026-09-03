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

namespace umc::comm {

#pragma pack(push, 1)
struct UbSignalValue {
    uint16_t magic;      // [0,2)  == kUbSignalMagicU16 (0x7A5A)
    uint8_t version;     // [2,3)  == kUbSignalVersionV1 (1)
    uint8_t payload[5];  // [3,8)
};
#pragma pack(pop)

static_assert(sizeof(UbSignalValue) == 8, "UbSignalValue must be 8 bytes");
static_assert(offsetof(UbSignalValue, magic) == 0, "");
static_assert(offsetof(UbSignalValue, version) == 2, "");
static_assert(offsetof(UbSignalValue, payload) == 3, "");

#pragma pack(push, 1)
struct UbSignalSlot {
    uint64_t signal;         // [0,8)
    uint8_t bizPayload[56];  // [8,64)
};
#pragma pack(pop)

static_assert(sizeof(UbSignalSlot) == 64, "UbSignalSlot must be 64 bytes (cacheline)");
static_assert(offsetof(UbSignalSlot, signal) == 0, "");
static_assert(offsetof(UbSignalSlot, bizPayload) == 8, "");

constexpr uint16_t kUbSignalMagicU16 = 0x7A5A;
constexpr uint64_t kUbSignalMagic = static_cast<uint64_t>(kUbSignalMagicU16);
constexpr uint64_t kUbSignalMagicMask = 0x000000000000FFFFULL;
constexpr uint8_t kUbSignalVersionV1 = 1;
constexpr uint8_t kUbSignalReserved = 0;

constexpr uint16_t ExtractMagic(uint64_t raw)
{
    return static_cast<uint16_t>(raw & kUbSignalMagicMask);
}

constexpr uint8_t ExtractVersion(uint64_t raw) { return static_cast<uint8_t>((raw >> 16) & 0xFFu); }

constexpr bool IsValidSignal(uint64_t raw)
{
    return ExtractMagic(raw) == kUbSignalMagicU16 && ExtractVersion(raw) == kUbSignalVersionV1;
}

}  // namespace umc::comm
