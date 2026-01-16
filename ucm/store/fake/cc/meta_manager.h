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
#ifndef UNIFIEDCACHE_FAKE_STORE_CC_META_MANAGER_H
#define UNIFIEDCACHE_FAKE_STORE_CC_META_MANAGER_H

#include <memory>
#include "global_config.h"
#include "status/status.h"
#include "type/types.h"

namespace UC::FakeStore {

class MetaStrategy;

class MetaManager {
    std::shared_ptr<MetaStrategy> strategy_{nullptr};

public:
    Status Setup(const Config& config);
    void Insert(const Detail::BlockId& block) noexcept;
    bool Exist(const Detail::BlockId& block) const noexcept;

private:
    bool ExistAt(size_t iBucket, const Detail::BlockId& block) const noexcept;
    void InsertAt(size_t iBucket, const Detail::BlockId& block) noexcept;
    void MoveTo(size_t iBucket, size_t iNode) noexcept;
    void Remove(size_t iBucket, size_t iNode) noexcept;
};

}  // namespace UC::FakeStore

#endif
