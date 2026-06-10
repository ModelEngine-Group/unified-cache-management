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
#ifndef AIO_HOST_BUFFER_H
#define AIO_HOST_BUFFER_H

#include <cstddef>
#include <cstdint>

namespace aio {

class HostBuffer {
public:
    enum class Strategy : uint8_t { ALLOC, MMAP };

    HostBuffer(Strategy strategy, int32_t deviceId, size_t size, size_t number);
    ~HostBuffer();
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    HostBuffer(HostBuffer&&) = delete;
    HostBuffer& operator=(HostBuffer&&) = delete;
    size_t Size() const { return size_; }
    size_t Number() const { return number_; }
    void* Buffer() const { return buffer_; }
    void* operator[](size_t index) const { return static_cast<char*>(buffer_) + index * size_; }

private:
    Strategy strategy_;
    int32_t deviceId_;
    size_t size_;
    size_t number_;
    void* buffer_;
};

}  // namespace aio

#endif  // AIO_HOST_BUFFER_H
