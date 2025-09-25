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

#include <list>
#include <string>
#include <cstdint>

namespace UC {

class CCStore {
    using BlockId = std::string;
    using TaskId = size_t;

public:
    static constexpr TaskId invalidTaskId = 0;
    class Task {
    public:
        struct Shard {
            size_t index;
            BlockId block;
            size_t offset;
            uintptr_t address;
        };
        enum class Type { DUMP, LOAD };
        enum class Location { HOST, DEVICE };
        Type type;
        Location location;
        std::string brief;
        size_t number;
        size_t size;
        std::list<Shard> shards;
        Task(const Type type, const Location location, const std::string& brief)
            : type{type}, location{location}, brief{brief}, number{0}, size{0}
        {
        }
        int32_t Append(const BlockId& block, const size_t offset, const uintptr_t address,
                       const size_t length)
        {
            if (this->number == 0) { this->size = length; }
            if (this->size != length) { return -1; }
            this->shards.emplace_back<Shard>({this->number, block, offset, address});
            this->number++;
            return 0;
        }
    };

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
