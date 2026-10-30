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
#include <dlfcn.h>

namespace UC::Trans::Hal {
namespace {

// libascend_hal.so belongs to the host driver, not to CANN. It is loaded at run time so
// the wheel does not depend on it and can be built on machines without the driver.
void* HalLibrary()
{
    static void* library = dlopen("libascend_hal.so", RTLD_NOW | RTLD_LOCAL);
    return library;
}

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

template <typename Fn, typename... Args>
Status CallHal(const char* api, Args... args)
{
    void* library = HalLibrary();
    void* symbol = library != nullptr ? dlsym(library, api) : nullptr;
    if (symbol != nullptr) {
        return FromHalError(api, reinterpret_cast<Fn>(symbol)(args...));
    } else {
        return Status::OsApiError(fmt::format("cannot load {} from libascend_hal.so", api));
    }
}

#define UC_HAL_CALL(fn, ...) CallHal<decltype(&fn)>(#fn, __VA_ARGS__)

}  // namespace

Status MemGetAllocationGranularity(PageType pageType, size_t* granularity)
{
    const drv_mem_prop prop = HostMemoryProp(pageType);
    return UC_HAL_CALL(halMemGetAllocationGranularity, &prop, MEM_ALLOC_GRANULARITY_RECOMMENDED,
                       granularity);
}

Status MemAddressReserve(void** ptr, size_t size)
{
    return UC_HAL_CALL(halMemAddressReserve, ptr, size, 0, nullptr, 0);
}

Status MemAddressFree(void* ptr) { return UC_HAL_CALL(halMemAddressFree, ptr); }

Status MemCreate(MemHandle* handle, size_t size, PageType pageType)
{
    const drv_mem_prop prop = HostMemoryProp(pageType);
    drv_mem_handle_t* nativeHandle = nullptr;
    Status status = UC_HAL_CALL(halMemCreate, &nativeHandle, size, &prop, 0);
    if (status.Success()) { *handle = nativeHandle; }
    return status;
}

Status MemRelease(MemHandle handle)
{
    return UC_HAL_CALL(halMemRelease, static_cast<drv_mem_handle_t*>(handle));
}

Status MemMap(void* ptr, size_t size, MemHandle handle)
{
    return UC_HAL_CALL(halMemMap, ptr, size, 0, static_cast<drv_mem_handle_t*>(handle), 0);
}

Status MemUnmap(void* ptr) { return UC_HAL_CALL(halMemUnmap, ptr); }

Status MemExportToShareableHandle(MemHandle handle, uint64_t* shareHandle)
{
    return UC_HAL_CALL(halMemExportToShareableHandle, static_cast<drv_mem_handle_t*>(handle),
                       MEM_HANDLE_TYPE_NONE, 0, shareHandle);
}

Status MemShareHandleDisableWhitelist(uint64_t shareHandle)
{
    ShareHandleAttr attr{};
    attr.enableFlag = SHR_HANDLE_NO_WLIST_ENABLE;
    return UC_HAL_CALL(halMemShareHandleSetAttribute, shareHandle,
                       SHR_HANDLE_ATTR_NO_WLIST_IN_SERVER, attr);
}

Status MemImportFromShareableHandle(uint64_t shareHandle, uint32_t deviceId, MemHandle* handle)
{
    drv_mem_handle_t* nativeHandle = nullptr;
    Status status =
        UC_HAL_CALL(halMemImportFromShareableHandle, shareHandle, deviceId, &nativeHandle);
    if (status.Success()) { *handle = nativeHandle; }
    return status;
}

}  // namespace UC::Trans::Hal
