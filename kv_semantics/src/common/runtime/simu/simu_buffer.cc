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
#include "simu_buffer.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace kv::runtime {

static void* AllocMemory(size_t size, int8_t initVal)
{
    auto ptr = malloc(size);
    if (!ptr) { return nullptr; }
    std::memset(ptr, initVal, size);
    return ptr;
}

static void FreeMemory(void* ptr) { free(ptr); }

std::shared_ptr<void> SimuBuffer::MakeDeviceMappedHostBuffer(size_t size)
{
    return MakeHostBuffer(size);
}

std::shared_ptr<void> SimuBuffer::MakeHostMappedDeviceBuffer(size_t size, void** pDevice)
{
    auto host = MakeHostBuffer(size);
    if (pDevice) { *pDevice = host.get(); }
    return host;
}

std::shared_ptr<void> SimuBuffer::MakeDeviceBuffer(size_t size)
{
    constexpr int8_t deviceInitVal = 0xd;
    auto device = AllocMemory(size, deviceInitVal);
    if (!device) { return nullptr; }
    return std::shared_ptr<void>(device, FreeMemory);
}

std::shared_ptr<void> SimuBuffer::MakeHostBuffer(size_t size)
{
    constexpr int8_t hostInitVal = 0xa;
    auto device = AllocMemory(size, hostInitVal);
    if (!device) { return nullptr; }
    return std::shared_ptr<void>(device, FreeMemory);
}

Status Memset(void* ptr, std::size_t size, std::int32_t value)
{
    std::memset(ptr, static_cast<int>(value), size);
    return Status::OK();
}

}  // namespace kv::runtime
