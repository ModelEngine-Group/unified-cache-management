
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

#include "device/idevice.h"

namespace UC {

class IBufferedDevice : public IDevice {  
    class Buffer{
    public:
        void Setup(std::shared_ptr<void> buffer, size_t size, size_t number)
        {
            this->_buffer = buffer;
            this->_size = size;
            this->_number = number;
            this->_index = 0;
        }
        bool Empty() const {return this->_index == 0;}
        bool Full() const {return this->_index == this->_number;}
        bool Available(size_t size) const {return this->_index <= this->_number;}
        void Reset() {this->_index = 0;}
        std::shared_ptr<void> GetBuffer()
        {
            auto ptr = static_cast<uint8_t*>(this->_buffer.get());
            auto buffer = ptr + this->_size * this->_index;
            this->_index++;
            return std::shared_ptr<void>((void*)buffer, [](void*){});
        }

    private:
        std::shared_ptr<void> _buffer{nullptr};
        size_t _size{0};
        size_t _number{0};
        size_t _index{0};
    };

public:
    IBufferedDevice(size_t bufferSize, size_t bufferNumber);
    virtual ~IBufferedDevice() = default;
    Status Setup() override;
    std::shared_ptr<void> GetHostBuffer(size_t size) override;

private:
    size_t _bufferSize;
    size_t _bufferNumber;
    std::shared_ptr<void> _tmpBuffer;
    Buffer _buffer;
};

} // namespace UC

#endif // UNIFIEDCACHE_IBUFFERED_DEVICE_H