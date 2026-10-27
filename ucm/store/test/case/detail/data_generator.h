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
#include <cstdlib>
#include <cstring>
#include <new>

namespace UC::Test::Detail {

class DataGenerator {
public:
    explicit DataGenerator(const size_t nPage, const size_t pageSize = 4096)
        : nPage_{nPage}, pageSize_{pageSize}, data_{nullptr}
    {
    }
    ~DataGenerator()
    {
        if (this->data_) {
            free(this->data_);
            this->data_ = nullptr;
        }
    }
    void Generate()
    {
        // Posix tests also use O_DIRECT, which requires an aligned buffer.
        void* fresh = nullptr;
        const auto bytes = this->Size();
        if (posix_memalign(&fresh, 4096, bytes == 0 ? 1 : bytes) != 0) {
            throw std::bad_alloc();
        }
        free(this->data_);
        this->data_ = fresh;
    }
    void GenerateRandom()
    {
        this->Generate();
        for (size_t i = 0; i < this->nPage_; i++) {
            *(size_t*)((char*)this->data_ + this->pageSize_ * i) = i;
        }
    }
    int32_t Compare(const DataGenerator& other)
    {
        if (this->nPage_ < other.nPage_) { return -1; }
        if (this->nPage_ < other.nPage_) { return 1; }
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
    size_t nPage_;
    size_t pageSize_;
    void* data_;
};

}  // namespace UC::Test::Detail

#endif
