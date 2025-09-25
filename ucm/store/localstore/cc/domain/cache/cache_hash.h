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
#ifndef UNIFIEDCACHE_CACHE_HASH_H
#define UNIFIEDCACHE_CACHE_HASH_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace UC {

class CacheHash {
    uint32_t capacity_;
    void* addr_;

public:
    static constexpr uint32_t npos = std::numeric_limits<uint32_t>::max();
    CacheHash() : capacity_{0}, addr_{nullptr} {}
    void Setup(const uint32_t capacity) noexcept { this->capacity_ = capacity; }
    size_t MemorySize() const noexcept;
    void Setup(void* addr) noexcept;
    void Insert(const std::string& id, const size_t offset, const uint32_t index) noexcept;
    uint32_t Find(const std::string& id, const size_t offset) noexcept;
    void PutRef(const uint32_t index) noexcept;
    void PutRef(const std::string& id, const size_t offset) noexcept;
    void Remove(const std::string& id, const size_t offset) noexcept;
    uint32_t Evict() noexcept;
};

} // namespace UC

#endif
