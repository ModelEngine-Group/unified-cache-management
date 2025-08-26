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
#ifndef UCM_LOCAL_STORE_LRU_CACHE_H
#define UCM_LOCAL_STORE_LRU_CACHE_H

#include <memory>
#include <string_view>
#include "status/status.h"
#include "file/file.h"
#include "index_queue/index_queue.h"

namespace UCM {

struct LRUCacheHeader;

class LRUCache {
public:
    LRUCache();
    ~LRUCache();

    Status Initialize(uint64_t cache_num, uint64_t cache_size);
    Status Insert(std::string_view key, void** val);
    Status Find(std::string_view key, void** val);
    void Done(void* val);

private:
    Status MappingCheck(uint64_t shm_cap);
    Status MappingInitialize(uint64_t shm_cap, uint64_t cache_num, uint64_t cache_size);
    void Remove();

    LRUCacheHeader* _h;
    std::unique_ptr<IFile> _f;
    IndexQueue _q;
};

} // namespace UCM

#endif // UCM_LOCAL_STORE_LRU_CACHE_H