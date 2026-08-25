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
#include "core/transport.h"

namespace transport {

namespace detail {

uint64_t PtrToU64(const void* ptr);
void* U64ToPtr(uint64_t value);

bool AppendU64(Metadata& out, uint64_t value);
bool ReadU64(const Metadata& input, size_t& offset, uint64_t& value);
bool AppendU32(Metadata& out, uint32_t value);
bool ReadU32(const Metadata& input, size_t& offset, uint32_t& value);
bool AppendU16(Metadata& out, uint16_t value);
bool ReadU16(const Metadata& input, size_t& offset, uint16_t& value);
bool AppendU8(Metadata& out, uint8_t value);
bool ReadU8(const Metadata& input, size_t& offset, uint8_t& value);
bool AppendBytes(Metadata& out, const Metadata& value);
bool ReadBytes(const Metadata& input, size_t& offset, Metadata& value);
bool AppendString(Metadata& out, const std::string& value);
bool ReadString(const Metadata& input, size_t& offset, std::string& value);

}  // namespace detail
}  // namespace transport
