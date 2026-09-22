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
#include "hal_memory.h"

namespace UC::Trans::Hal {
namespace {

Status FromHalError(const char* api, drvError_t ret)
{
    if (ret == DRV_ERROR_NONE) { return Status::OK(); }
    return {static_cast<int32_t>(ret),
            fmt::format("{} failed: ret={}", api, static_cast<int>(ret))};
}

}  // namespace

Status MemGetAllocationGranularity(const drv_mem_prop* prop, drv_mem_granularity_options option,
                                   size_t* granularity)
{
    return FromHalError("halMemGetAllocationGranularity",
                        halMemGetAllocationGranularity(prop, option, granularity));
}

Status MemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr, uint64_t flags)
{
    return FromHalError("halMemAddressReserve",
                        halMemAddressReserve(ptr, size, alignment, addr, flags));
}

Status MemAddressFree(void* ptr)
{
    return FromHalError("halMemAddressFree", halMemAddressFree(ptr));
}

Status MemCreate(drv_mem_handle_t** handle, size_t size, const drv_mem_prop* prop, uint64_t flags)
{
    return FromHalError("halMemCreate", halMemCreate(handle, size, prop, flags));
}

Status MemRelease(drv_mem_handle_t* handle)
{
    return FromHalError("halMemRelease", halMemRelease(handle));
}

Status MemMap(void* ptr, size_t size, size_t offset, drv_mem_handle_t* handle, uint64_t flags)
{
    return FromHalError("halMemMap", halMemMap(ptr, size, offset, handle, flags));
}

Status MemUnmap(void* ptr) { return FromHalError("halMemUnmap", halMemUnmap(ptr)); }

Status MemExportToShareableHandle(drv_mem_handle_t* handle, drv_mem_handle_type type,
                                  uint64_t flags, uint64_t* shareHandle)
{
    return FromHalError("halMemExportToShareableHandle",
                        halMemExportToShareableHandle(handle, type, flags, shareHandle));
}

Status MemShareHandleSetAttribute(uint64_t shareHandle, ShareHandleAttrType type,
                                  ShareHandleAttr attr)
{
    return FromHalError("halMemShareHandleSetAttribute",
                        halMemShareHandleSetAttribute(shareHandle, type, attr));
}

Status MemImportFromShareableHandle(uint64_t shareHandle, uint32_t deviceId,
                                    drv_mem_handle_t** handle)
{
    return FromHalError("halMemImportFromShareableHandle",
                        halMemImportFromShareableHandle(shareHandle, deviceId, handle));
}

}  // namespace UC::Trans::Hal
