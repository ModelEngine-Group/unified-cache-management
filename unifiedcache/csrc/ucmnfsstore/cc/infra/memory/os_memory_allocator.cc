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
#include "os_memory_allocator.h"
#include <cstdlib>
#include <cstring>

namespace UC {

void* OsMemoryAlloc(const size_t size, bool initZero)
{
    return initZero ? std::calloc(1, size) : std::malloc(size);
}

void* OsMemoryAllocAlign(const size_t size, const size_t& bsSize, bool initZero)
{
    void* ptr = nullptr;
    auto ret = posix_memalign(&ptr, bsSize, size);
    if (ret != 0 || ptr == nullptr) {
        return nullptr;
    }
    if (initZero) {
        (void)std::memset(ptr, 0, size);
    }
    return ptr;
}

void OsMemoryFree(void* ptr)
{
    free(ptr);
}

std::shared_ptr<void> OsMemoryAllocator::Alloc(const size_t size, bool initZero)
{
    auto ptr = OsMemoryAlloc(size, initZero);
    if (ptr == nullptr) {
        return nullptr;
    }
    return std::shared_ptr<void>(ptr, OsMemoryFree);
}

bool OsMemoryAllocator::Aligned(const size_t& size)
{
    return (size % this->_alignment) == 0;
}

size_t OsMemoryAllocator::Align(const size_t size)
{
    return (size + this->_alignment - 1) / this->_alignment * this->_alignment;
}

std::shared_ptr<void> OsMemoryAllocator::AllocAligned(const size_t size, bool initZero)
{
    if (!this->Aligned(size)) {
        return nullptr;
    }
    auto ptr = OsMemoryAllocAlign(size, this->_alignment, initZero);
    if (ptr == nullptr) {
        return nullptr;
    }
    return std::shared_ptr<void>(ptr, OsMemoryFree);
}

} // namespace UC