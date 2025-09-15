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
#include "ibuffered_device.h"
#include "logger/logger.h"

namespace UC {

Status IBufferedDevice::Setup()
{
    auto totalSize = this->_bufferSize * this->_bufferNumber;
    this->_addr = this->MakeBuffer(totalSize);
    if (!this->_addr) { return Status::OutOfMemory(); }
    this->_indexPool.Setup(this->_bufferNumber);
    return Status::OK();
}

std::shared_ptr<std::byte> IBufferedDevice::GetBuffer(const size_t size)
{
    auto idx = IndexPool::npos;
    if (size <= this->_bufferSize && (idx = this->_indexPool.Acquire()) != IndexPool::npos) {
        auto ptr = this->_addr.get() + this->_bufferSize * idx;
        return std::shared_ptr<std::byte>(ptr, [this, idx](std::byte*) { this->_indexPool.Release(idx); });
    }
    auto buffer = this->MakeBuffer(size);
    if (buffer) { return buffer; }
    UC_WARN("Reservation exhausted, try to alloc OS memory({}).", size);
    auto host = (std::byte*)malloc(size);
    if (host) { return std::shared_ptr<std::byte>(host, free); }
    return nullptr;
}

} // namespace UC
