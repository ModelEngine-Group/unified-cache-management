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
#ifndef UNIFIEDCACHE_TEST_DETAIL_DATA_GENERATOR_H
#define UNIFIEDCACHE_TEST_DETAIL_DATA_GENERATOR_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace UC::Test::Detail {

class DataGenerator {
public:
    // Keep in sync with the I/O alignment used by the store layer.
    static constexpr size_t kIoAlignment = 4096;

    explicit DataGenerator(const size_t nPage, const size_t pageSize = 4096)
        : nPage_{nPage}, pageSize_{pageSize}, data_{nullptr}
    {
    }
    ~DataGenerator() { FreeBuffer(); }

    DataGenerator(const DataGenerator&) = delete;
    DataGenerator& operator=(const DataGenerator&) = delete;

    DataGenerator(DataGenerator&& other) noexcept
        : nPage_{other.nPage_}, pageSize_{other.pageSize_}, data_{other.data_}
    {
        other.data_ = nullptr;
    }
    DataGenerator& operator=(DataGenerator&& other) noexcept
    {
        if (this != &other) {
            FreeBuffer();
            nPage_ = other.nPage_;
            pageSize_ = other.pageSize_;
            data_ = other.data_;
            other.data_ = nullptr;
        }
        return *this;
    }

    void Generate()
    {
        FreeBuffer();

        const size_t bytes = this->Size();
        if (bytes == 0) { return; }

        const size_t rounded = RoundUp(bytes, kIoAlignment);
        this->data_ = AlignedAlloc(kIoAlignment, rounded);
        if (this->data_ == nullptr) {
            throw std::bad_alloc();
        }
    }
    void GenerateRandom()
    {
        this->Generate();
        for (size_t i = 0; i < this->nPage_; i++) {
            *(size_t*)((char*)this->data_ + this->pageSize_ * i) = i;
        }
    }
    int32_t Compare(const DataGenerator& other) const
    {
        if (this->nPage_ < other.nPage_) { return -1; }
        if (this->nPage_ > other.nPage_) { return 1; }
        for (size_t i = 0; i < this->nPage_; i++) {
            auto ret = memcmp((char*)this->data_ + this->pageSize_ * i,
                              (char*)other.data_ + this->pageSize_ * i, this->pageSize_);
            if (ret != 0) { return ret; }
        }
        return 0;
    }
    size_t Size() const { return this->pageSize_ * this->nPage_; }
    void* Buffer() const { return this->data_; }

private:
    static size_t RoundUp(const size_t value, const size_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    static void* AlignedAlloc(const size_t alignment, const size_t size)
    {
#if defined(_WIN32)
        return _aligned_malloc(size, alignment);
#else
        void* ptr = nullptr;
        if (::posix_memalign(&ptr, alignment, size) != 0) { return nullptr; }
        return ptr;
#endif
    }

    static void AlignedFree(void* ptr)
    {
        if (ptr == nullptr) { return; }
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        ::free(ptr);
#endif
    }

    void FreeBuffer()
    {
        if (this->data_) {
            AlignedFree(this->data_);
            this->data_ = nullptr;
        }
    }

    size_t nPage_;
    size_t pageSize_;
    void* data_;
};

}  // namespace UC::Test::Detail

#endif