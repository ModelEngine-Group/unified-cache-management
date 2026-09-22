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
#include "status/status.h"

namespace UC::Trans::Hal {

using MemHandle = void*;

enum class PageType : uint32_t { Normal, Huge };

// HAL reservations larger than 512 MiB require 1 GiB alignment.
inline constexpr size_t kAddressAlignment = size_t{1} << 30;

// Each wrapper forwards one HAL call and preserves the driver error code.
// Allocations use host DDR; the granularity query returns the recommended size.
Status MemGetAllocationGranularity(PageType pageType, size_t* granularity);
Status MemAddressReserve(void** ptr, size_t size);
Status MemAddressFree(void* ptr);
Status MemCreate(MemHandle* handle, size_t size, PageType pageType);
Status MemRelease(MemHandle handle);
Status MemMap(void* ptr, size_t size, MemHandle handle);
Status MemUnmap(void* ptr);
Status MemExportToShareableHandle(MemHandle handle, uint64_t* shareHandle);
Status MemShareHandleDisableWhitelist(uint64_t shareHandle);
Status MemImportFromShareableHandle(uint64_t shareHandle, uint32_t deviceId, MemHandle* handle);

}  // namespace UC::Trans::Hal
