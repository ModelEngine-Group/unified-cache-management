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
#ifndef UNIFIEDCACHE_MOONCAKE_STORE_CC_TRANS_TASK_H
#define UNIFIEDCACHE_MOONCAKE_STORE_CC_TRANS_TASK_H

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>
#include "type/types.h"

namespace UC::MooncakeStore {

enum class TaskType { LOAD, DUMP };

struct TransShard {
    std::string key;
    Detail::BlockId owner;
    size_t index;
    std::vector<void*> addrs;
    std::vector<size_t> sizes;
    std::shared_ptr<void> cacheLifeHolder;
};

struct TransTask {
    Detail::TaskHandle id{NextId()};
    TaskType type{TaskType::DUMP};
    std::string brief;
    std::vector<TransShard> shards;
    uintptr_t prerequisiteHandle{0};

private:
    static Detail::TaskHandle NextId() noexcept
    {
        static std::atomic<Detail::TaskHandle> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    }
};

}  // namespace UC::MooncakeStore

#endif
