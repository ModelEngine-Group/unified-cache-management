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
#include "cache_instance.h"

namespace UC {

Status CacheInstance::Setup(const size_t ioSize, const uint32_t capacity) noexcept
{
    auto status = this->meta_.Setup(capacity);
    if (status.Failure()) { return status; }
    return this->data_.Setup(ioSize, capacity);
}

void* CacheInstance::Alloc(const std::string& id, const size_t offset) noexcept
{
    auto index = this->meta_.Alloc(id, offset);
    return index != this->meta_.npos ? this->data_.At(index) : nullptr;
}

void* CacheInstance::Find(const std::string& id, const size_t offset) noexcept
{
    auto index = this->meta_.Find(id, offset);
    return index != this->meta_.npos ? this->data_.At(index) : nullptr;
}

void CacheInstance::PutRef(const std::string& id, const size_t offset) noexcept
{
    this->meta_.PutRef(id, offset);
}

} // namespace UC
