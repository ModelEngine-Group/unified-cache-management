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
#include "token_codec.h"

namespace UC::Store {
namespace {

void PutU32(std::byte* dst, std::uint32_t value)
{
    for (int shift = 0; shift <= 24; shift += 8) {
        dst[shift / 8] = static_cast<std::byte>((value >> shift) & 0xFF);
    }
}

std::uint32_t GetU32(const std::byte* src)
{
    std::uint32_t value = 0;
    for (int shift = 0; shift <= 24; shift += 8) {
        value |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(src[shift / 8])) << shift;
    }
    return value;
}

void PutU64(std::byte* dst, std::uint64_t value)
{
    for (int shift = 0; shift <= 56; shift += 8) {
        dst[shift / 8] = static_cast<std::byte>((value >> shift) & 0xFF);
    }
}

std::uint64_t GetU64(const std::byte* src)
{
    std::uint64_t value = 0;
    for (int shift = 0; shift <= 56; shift += 8) {
        value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(src[shift / 8])) << shift;
    }
    return value;
}

}  // namespace

std::vector<std::byte> EncodeTokenPayload(const std::vector<std::uint32_t>& slots)
{
    std::vector<std::byte> data(slots.size() * 4);
    for (std::size_t i = 0; i < slots.size(); ++i) { PutU32(data.data() + i * 4, slots[i]); }
    return data;
}

std::vector<std::uint32_t> DecodeTokenPayload(const std::vector<std::byte>& data)
{
    std::vector<std::uint32_t> slots(data.size() / 4);
    for (std::size_t i = 0; i < slots.size(); ++i) { slots[i] = GetU32(data.data() + i * 4); }
    return slots;
}

std::vector<std::byte> EncodePlanPayload(const PlanPayload& plan)
{
    std::vector<std::byte> data(8 + plan.slots.size() * 4);
    PutU64(data.data(), plan.batchId);
    for (std::size_t i = 0; i < plan.slots.size(); ++i) {
        PutU32(data.data() + 8 + i * 4, plan.slots[i]);
    }
    return data;
}

PlanPayload DecodePlanPayload(const std::vector<std::byte>& data)
{
    PlanPayload plan;
    if (data.size() < 8) { return plan; }
    plan.batchId = GetU64(data.data());
    plan.slots.resize((data.size() - 8) / 4);
    for (std::size_t i = 0; i < plan.slots.size(); ++i) {
        plan.slots[i] = GetU32(data.data() + 8 + i * 4);
    }
    return plan;
}

}  // namespace UC::Store
