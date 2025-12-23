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
#ifndef UNIFIEDCACHE_TRANS_RESERVED_BUFFER_H
#define UNIFIEDCACHE_TRANS_RESERVED_BUFFER_H

#include <fmt/format.h>
#include "indexer.h"
#include "trans/buffer.h"

namespace UC::Trans {

class ReservedBuffer : public Buffer {
    struct {
        Indexer indexer;
        std::shared_ptr<void> buffers;
        size_t size;
    } hostBuffers_, deviceBuffers_;

    template <typename Buffers>
    static std::shared_ptr<void> GetBufferFrom(Buffers& buffers)
    {
        auto pos = buffers.indexer.Acquire();
        if (pos != buffers.indexer.npos) {
            auto addr = static_cast<int8_t*>(buffers.buffers.get());
            auto ptr = static_cast<void*>(addr + buffers.size * pos);
            return std::shared_ptr<void>(ptr,
                                         [&buffers, pos](void*) { buffers.indexer.Release(pos); });
        }
        return nullptr;
    }

public:
    Status MakeDeviceBuffers(size_t size, size_t number) override
    {
        auto totalSize = size * number;
        auto buffers = this->MakeDeviceBuffer(totalSize);
        if (!buffers) {
            return Status::Error(fmt::format("out of memory({}) on device", totalSize));
        }
        this->deviceBuffers_.size = size;
        this->deviceBuffers_.buffers = buffers;
        this->deviceBuffers_.indexer.Setup(number);
        return Status::OK();
    }

    std::shared_ptr<void> GetDeviceBuffer(size_t size) override
    {
        if (size <= this->deviceBuffers_.size) {
            auto buffer = GetBufferFrom(this->deviceBuffers_);
            if (buffer) { return buffer; }
        }
        return this->MakeDeviceBuffer(size);
    }

    Status MakeHostBuffers(size_t size, size_t number) override
    {
        auto totalSize = size * number;
        auto buffers = this->MakeHostBuffer(totalSize);
        if (!buffers) { return Status::Error(fmt::format("out of memory({}) on host", totalSize)); }
        this->hostBuffers_.size = size;
        this->hostBuffers_.buffers = buffers;
        this->hostBuffers_.indexer.Setup(number);
        return Status::OK();
    }

    std::shared_ptr<void> GetHostBuffer(size_t size) override
    {
        if (size <= this->hostBuffers_.size) {
            auto buffer = GetBufferFrom(this->hostBuffers_);
            if (buffer) { return buffer; }
        }
        return this->MakeHostBuffer(size);
    }
};

} // namespace UC::Trans

#endif
