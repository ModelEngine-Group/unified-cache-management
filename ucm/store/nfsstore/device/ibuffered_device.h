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
#ifndef UNIFIEDCACHE_IBUFFERED_DEVICE_H
#define UNIFIEDCACHE_IBUFFERED_DEVICE_H

#include "idevice.h"

namespace UC {

class IBufferedDevice : public IDevice {
    class LinearBuffer {
        std::shared_ptr<std::byte> addr_{nullptr};
        size_t index_{0};
        size_t number_{0};
        size_t size_{0};

    public:
        void Setup(std::shared_ptr<std::byte> addr, const size_t number, const size_t size)
        {
            this->addr_ = addr;
            this->number_ = number;
            this->size_ = size;
            this->Reset();
        }
        void Reset() noexcept { this->index_ = 0; }
        bool Full() const noexcept { return this->index_ == this->number_; }
        bool Available(const size_t size) const noexcept { return this->size_ >= size; }
        std::shared_ptr<std::byte> Get() noexcept
        {
            auto addr = this->addr_.get();
            auto buffer = addr + this->size_ * this->index_;
            ++this->index_;
            return std::shared_ptr<std::byte>(buffer, [](auto) {});
        }
    };
    LinearBuffer buffer_;

public:
    IBufferedDevice(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
        : IDevice{deviceId, bufferSize, bufferNumber}
    {
    }
    Status Setup() override
    {
        auto totalSize = this->bufferSize * this->bufferNumber;
        if (totalSize == 0) { return Status::OK(); }
        auto addr = this->MakeBuffer(totalSize);
        if (!addr) { return Status::OutOfMemory(); }
        this->buffer_.Setup(addr, this->bufferNumber, this->bufferSize);
        return Status::OK();
    }
    virtual std::shared_ptr<std::byte> GetBuffer(const size_t size) override
    {
        if (this->buffer_.Full()) {
            auto status = this->Synchronized();
            if (status.Failure()) { return nullptr; }
            this->buffer_.Reset();
        }
        return this->buffer_.Available(size) ? this->buffer_.Get() : this->MakeBuffer(size);
    }
};

} // namespace UC

#endif
