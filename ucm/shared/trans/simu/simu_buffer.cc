/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include "trans/buffer.h"

namespace UC::Trans {

static void* AllocMemory(size_t size, int8_t initVal)
{
    auto ptr = malloc(size);
    if (!ptr) { return nullptr; }
    std::memset(ptr, initVal, size);
    return ptr;
}

static void FreeMemory(void* ptr) { free(ptr); }

template <typename Buffers>
static std::shared_ptr<void> GetBuffer(Buffers& buffers)
{
    auto pos = buffers.indexer.Acquire();
    if (pos != buffers.indexer.npos) {
        auto addr = static_cast<int8_t*>(buffers.buffers.get());
        auto ptr = static_cast<void*>(addr + buffers.size * pos);
        return std::shared_ptr<void>(ptr, [&buffers, pos](void*) { buffers.indexer.Release(pos); });
    }
    return nullptr;
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

Status SimuBuffer::RegisterHostBuffer(void* ptr, size_t size) { return Status::OK(); }

void SimuBuffer::UnregisterHostBuffer(void* ptr) {}

void* SimuBuffer::GetHostPtrOnDevice(void* ptr) { return ptr; }

} // namespace UC::Trans
