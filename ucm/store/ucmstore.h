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
#ifndef UNIFIEDCACHE_STORE_H
#define UNIFIEDCACHE_STORE_H

#include <cstddef>
#include <list>
#include <string>

namespace UC {

template <class Task>
class CCStore {
    using BlockId = std::string;
    using TaskId = size_t;

public:
    virtual ~CCStore() = default;
    virtual int32_t Alloc(const BlockId& block) = 0;
    virtual bool Lookup(const BlockId& block) = 0;
    virtual void Commit(const BlockId& block, const bool success) = 0;
    virtual std::list<int32_t> Alloc(const std::list<BlockId>& blocks) = 0;
    virtual std::list<bool> Lookup(const std::list<BlockId>& blocks) = 0;
    virtual void Commit(const std::list<BlockId>& blocks, const bool success) = 0;
    virtual TaskId Submit(Task&& task) = 0;
    virtual int32_t Wait(const TaskId task) = 0;
    virtual int32_t Check(const TaskId task, bool& finish) = 0;
};

} // namespace UC

#endif
