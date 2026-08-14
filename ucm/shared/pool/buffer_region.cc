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
#include "pool/buffer_region.h"
#include <utility>
#include "trans/device.h"

namespace UC {

Status BufferRegion::Create(BufferMemoryType type, std::size_t size, BufferRegion& region)
{
    auto buffer = Trans::Device{}.MakeBuffer();
    if (!buffer) { return Status::Error("failed to create runtime buffer"); }

    switch (type) {
        case BufferMemoryType::HOST: {
            auto owner = buffer->MakeHostBuffer(size);
            if (!owner) { return Status::Error("failed to allocate host memory"); }
            region.owner = std::move(owner);
            region.local_addr = region.owner.get();
            region.device_addr = region.owner.get();
            return Status::OK();
        }
        case BufferMemoryType::HOST_PINNED: {
            void* deviceAddr = nullptr;
            auto owner = buffer->MakeHostPinnedBuffer(size, &deviceAddr);
            if (!owner || !deviceAddr) {
                return Status::Error("failed to allocate host-pinned memory");
            }
            region.owner = std::move(owner);
            region.local_addr = region.owner.get();
            region.device_addr = deviceAddr;
            return Status::OK();
        }
        case BufferMemoryType::ASCEND_DEVICE: {
            auto owner = buffer->MakeDeviceBuffer(size);
            if (!owner) { return Status::Error("failed to allocate device memory"); }
            region.owner = std::move(owner);
            region.local_addr = region.owner.get();
            region.device_addr = region.owner.get();
            return Status::OK();
        }
        case BufferMemoryType::DEVICE_MAPPED_HOST: {
            auto owner = buffer->MakeCpuAccessibleDeviceBuffer(size);
            if (!owner) { return Status::Error("failed to allocate CPU-accessible device memory"); }
            region.owner = std::move(owner);
            region.local_addr = region.owner.get();
            region.device_addr = region.owner.get();
            return Status::OK();
        }
        default: return Status::InvalidParam("unsupported memory type");
    }
}

void BufferRegion::Reset()
{
    owner.reset();
    local_addr = nullptr;
    device_addr = nullptr;
}

}  // namespace UC
