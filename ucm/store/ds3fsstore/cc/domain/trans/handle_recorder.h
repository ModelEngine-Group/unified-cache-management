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
#ifndef UC_INFRA_HANDLE_POOL_H
#define UC_INFRA_HANDLE_POOL_H

#include <functional>
#include "hashmap.h"
#include "status/status.h"

namespace UC {

template <typename KeyType, typename HandleType>
class HandlePool {
private:
    struct PoolEntry {
        HandleType handle;
        uint64_t refCount;
    };
    using PoolMap = HashMap<KeyType, PoolEntry, std::hash<KeyType>, 10>;
    PoolMap pool_;

public:
    HandlePool() = default;
    HandlePool(const HandlePool&) = delete;
    HandlePool& operator=(const HandlePool&) = delete;

    static HandlePool& Instance()
    {
        static HandlePool instance;
        return instance;
    }

    Status Get(const KeyType& key, HandleType& handle,
               std::function<Status(HandleType&)> instantiate)
    {
        auto result = pool_.GetOrCreate(key, [&instantiate](PoolEntry& entry) -> bool {
            HandleType h{};

            auto status = instantiate(h);
            if (status.Failure()) { return false; }

            entry.handle = h;
            entry.refCount = 1;
            return true;
        });

        if (!result.has_value()) { return Status::Error(); }

        auto& entry = result.value().get();
        entry.refCount++;
        handle = entry.handle;
        return Status::OK();
    }

    void Put(const KeyType& key, std::function<void(HandleType)> cleanup)
    {
        pool_.Upsert(key, [&cleanup](PoolEntry& entry) -> bool {
            entry.refCount--;
            if (entry.refCount > 0) { return false; }
            cleanup(entry.handle);
            return true;
        });
    }

    void ClearAll(std::function<void(HandleType)> cleanup)
    {
        pool_.ForEach([&cleanup](const KeyType& key, PoolEntry& entry) {
            (void)key;
            cleanup(entry.handle);
        });
        pool_.Clear();
    }
};

} // namespace UC

#endif
