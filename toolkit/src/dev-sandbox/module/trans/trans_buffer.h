/**
 * MIT License
 *
 * Copyright (c) 2026 Mag1c.H
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
#ifndef TRANS_BUFFER_H
#define TRANS_BUFFER_H

#include <string>

class TransBuffer {
protected:
    std::size_t device_;
    std::size_t size_;
    std::size_t number_;
    void* addr_;

public:
    TransBuffer(std::size_t device, std::size_t size, std::size_t number)
        : device_{device}, size_{size}, number_{number}, addr_{nullptr}
    {
    }
    virtual ~TransBuffer() = default;
    virtual std::string Name() const = 0;
    const std::size_t& Device() const { return device_; }
    const std::size_t& Size() const { return size_; }
    const std::size_t& Number() const { return number_; }
    void* operator[](std::size_t i) const
    {
        return static_cast<void*>(static_cast<char*>(addr_) + i * size_);
    }
};

#endif  // TRANS_BUFFER_H
