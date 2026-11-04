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
#include "host_buffer.h"
#include <cstdlib>
#include <sys/mman.h>
#include "error_handle.h"

namespace aio {

HostBuffer::HostBuffer(Strategy strategy, int32_t deviceId, size_t size, size_t number)
    : strategy_(strategy), deviceId_(deviceId), size_(size), number_(number), buffer_(nullptr)
{
    const auto totalSize = size_ * number_;
    if (strategy == Strategy::ALLOC) {
        const auto alignment = 4096;
        AIO_ASSERT(posix_memalign(&buffer_, alignment, totalSize) == 0);
    } else {
        buffer_ = mmap(nullptr, totalSize, PROT_READ | PROT_WRITE,
                       MAP_ANONYMOUS | MAP_PRIVATE | MAP_POPULATE, -1, 0);
        AIO_ASSERT(buffer_ != MAP_FAILED);
    }
}

HostBuffer::~HostBuffer()
{
    if (!buffer_) { return; }
    if (strategy_ == Strategy::ALLOC) {
        free(buffer_);
    } else {
        munmap(buffer_, size_ * number_);
    }
    buffer_ = nullptr;
}

}  // namespace aio
