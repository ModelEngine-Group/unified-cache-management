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
#include "buffer.h"
#include "helper.h"

namespace UC {

Status Buffer::Setup(const size_t size, const size_t number)
{
    auto totalSize = size * number;
    if (totalSize == 0) { return Status::Error(); }
    this->base_ = this->MakeBuffer(totalSize);
    if (!this->base_) { return Status::OutOfMemory(); }
    this->index_.Setup(number);
    this->size_ = size;
    return Status::OK();
}

std::shared_ptr<void> Buffer::GetBuffer(const size_t size)
{
    if (size > this->size_) {
        UC_WARN("Buffer size({}) is not as expected({}).", size, this->size_);
        return this->MakeBuffer(size);
    }
    auto idx = this->index_.Acquire();
    if (idx != IndexPool::npos) {
        auto ptr = (void*)(((std::byte*)this->base_.get()) + this->size_ * idx);
        return std::shared_ptr<void>(ptr, [this, idx](auto) { this->index_.Release(idx); });
    }
    UC_WARN("All reserved buffer has been exhausted.");
    return this->MakeBuffer(size);
}

std::shared_ptr<void> Buffer::MakeBuffer(const size_t size)
{
    void* hostAddr = nullptr;
    auto ret = cudaHostAlloc(&hostAddr, size, cudaHostAllocPortable);
    if (ret != cudaSuccess) {
        CUDA_ERROR(ret);
        return nullptr;
    }
    return std::shared_ptr<void>(hostAddr, [](void* ptr) {
        auto ret = cudaFreeHost(ptr);
        if (ret != cudaSuccess) { CUDA_ERROR(ret); }
    });
}

} // namespace UC
