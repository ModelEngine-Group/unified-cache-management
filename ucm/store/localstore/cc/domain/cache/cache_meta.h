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
#ifndef UNIFIEDCACHE_CACHE_META_H
#define UNIFIEDCACHE_CACHE_META_H

#include "cache_hash.h"
#include "cache_index.h"
#include "file/ifile.h"
#include "status/status.h"

namespace UC {

class CacheMeta {
    using Index = uint32_t;
    void* addr_;
    size_t size_;
    CacheIndex index_;
    CacheHash hash_;

public:
    static constexpr Index npos = std::numeric_limits<Index>::max();
    CacheMeta() : addr_{nullptr}, size_{0} {}
    ~CacheMeta();
    Status Setup(const Index capacity) noexcept;
    Index Alloc(const std::string& id, const size_t offset) noexcept;
    Index Find(const std::string& id, const size_t offset) noexcept;
    void PutRef(const uint32_t index) noexcept;
    void PutRef(const std::string& id, const size_t offset) noexcept;

private:
    Status InitShmMeta(IFile* shmMetaFile);
    Status LoadShmMeta(IFile* shmMetaFile);
};

} // namespace UC

#endif
