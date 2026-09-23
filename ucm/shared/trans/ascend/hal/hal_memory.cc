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
#include <ascend_hal.h>

namespace UC::Trans::Hal {
namespace {

drv_mem_prop HostMemoryProp(PageType pageType)
{
    drv_mem_prop prop{};
    prop.side = MEM_HOST_SIDE;
    prop.devid = 0;
    prop.pg_type = pageType == PageType::Huge ? MEM_HUGE_PAGE_TYPE : MEM_NORMAL_PAGE_TYPE;
    prop.mem_type = MEM_DDR_TYPE;
    return prop;
}

Status FromHalError(const char* api, drvError_t ret)
{
    if (ret == DRV_ERROR_NONE) { return Status::OK(); }
    return {static_cast<int32_t>(ret),
            fmt::format("{} failed: ret={}", api, static_cast<int>(ret))};
}

}  // namespace

Status MemGetAllocationGranularity(PageType pageType, size_t* granularity)
{
    const drv_mem_prop prop = HostMemoryProp(pageType);
    return FromHalError(
        "halMemGetAllocationGranularity",
        halMemGetAllocationGranularity(&prop, MEM_ALLOC_GRANULARITY_RECOMMENDED, granularity));
}

Status MemAddressReserve(void** ptr, size_t size)
{
    return FromHalError("halMemAddressReserve", halMemAddressReserve(ptr, size, 0, nullptr, 0));
}

Status MemAddressFree(void* ptr)
{
    return FromHalError("halMemAddressFree", halMemAddressFree(ptr));
}

Status MemCreate(MemHandle* handle, size_t size, PageType pageType)
{
    const drv_mem_prop prop = HostMemoryProp(pageType);
    drv_mem_handle_t* nativeHandle = nullptr;
    Status status = FromHalError("halMemCreate", halMemCreate(&nativeHandle, size, &prop, 0));
    if (status.Success()) { *handle = nativeHandle; }
    return status;
}

Status MemRelease(MemHandle handle)
{
    return FromHalError("halMemRelease", halMemRelease(static_cast<drv_mem_handle_t*>(handle)));
}

Status MemMap(void* ptr, size_t size, MemHandle handle)
{
    return FromHalError("halMemMap",
                        halMemMap(ptr, size, 0, static_cast<drv_mem_handle_t*>(handle), 0));
}

Status MemUnmap(void* ptr) { return FromHalError("halMemUnmap", halMemUnmap(ptr)); }

Status MemExportToShareableHandle(MemHandle handle, uint64_t* shareHandle)
{
    return FromHalError("halMemExportToShareableHandle",
                        halMemExportToShareableHandle(static_cast<drv_mem_handle_t*>(handle),
                                                      MEM_HANDLE_TYPE_NONE, 0, shareHandle));
}

Status MemShareHandleDisableWhitelist(uint64_t shareHandle)
{
    ShareHandleAttr attr{};
    attr.enableFlag = SHR_HANDLE_NO_WLIST_ENABLE;
    return FromHalError(
        "halMemShareHandleSetAttribute",
        halMemShareHandleSetAttribute(shareHandle, SHR_HANDLE_ATTR_NO_WLIST_IN_SERVER, attr));
}

Status MemImportFromShareableHandle(uint64_t shareHandle, uint32_t deviceId, MemHandle* handle)
{
    drv_mem_handle_t* nativeHandle = nullptr;
    Status status =
        FromHalError("halMemImportFromShareableHandle",
                     halMemImportFromShareableHandle(shareHandle, deviceId, &nativeHandle));
    if (status.Success()) { *handle = nativeHandle; }
    return status;
}

}  // namespace UC::Trans::Hal
