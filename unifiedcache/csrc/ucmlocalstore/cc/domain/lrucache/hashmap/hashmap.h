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
#ifndef UCM_LOCAL_STORE_HASHMAP_H
#define UCM_LOCAL_STORE_HASHMAP_H

#include <string_view>
#include <optional>
#include "status/status.h"
#include "file/file.h"
#include "../index_queue/index_queue.h"

namespace UCM {

struct HashMapHeader;

class HashMap {
public:
    HashMap(std::string_view filename);
    ~HashMap();

    Status Initialize(const uint32_t map_size);
    Status Insert(std::string_view key, const uint32_t val);
    std::optional<uint32_t> Find(std::string_view key);
    void Remove(std::string_view key);

private:
    Status MappingCheck(const uint64_t shm_size);
    Status MappingInitialize(const uint64_t shm_size, const uint32_t map_size);

private:
    HashMapHeader* _h;
    std::unique_ptr<IFile> _f;
    IndexQueue _q;
};

} // namespace UCM

#endif // UCM_LOCAL_STORE_HASHMAP_H