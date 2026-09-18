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
#include <memory>
#include <sys/types.h>
#include "cache_buffer.h"
#include "ucmstore_v1.h"

namespace UC::Cache2 {

class BufferManager {
    std::unique_ptr<Buffer> buffer_;
    StoreV1* backend_{nullptr};

public:
    Status Setup(const Config& config) { return Status::Unsupported(); }
    Buffer* GetTransBuffer() { return buffer_ ? buffer_.get() : nullptr; }
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num)
    {
        return Status::Unsupported();
    }
    Expected<ssize_t> LookupOnReverse(const Detail::BlockId* blocks, size_t num)
    {
        return Status::Unsupported();
    }
    void Prefetch(const Detail::BlockId* blocks, size_t num)
    {
        if (buffer_) { buffer_->Touch(blocks, num); }
        if (backend_) { backend_->Prefetch(blocks, num); }
    }
};

}  // namespace UC::Cache2
