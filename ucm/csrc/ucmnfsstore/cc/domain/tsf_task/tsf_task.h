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
#ifndef UNIFIEDCACHE_TSF_TASK_H
#define UNIFIEDCACHE_TSF_TASK_H

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include "status/status.h"
#include "tsf_task_waiter.h"

namespace UC {

class TsfTask {
public:
    struct Shard {
        size_t index;
        std::string blockId;
        size_t offset;
        uintptr_t address;
    };
    enum class Type { DUMP, LOAD };
    enum class Location { HOST, DEVICE };
    Type type;
    Location location;
    std::string brief;
    size_t id;
    size_t number;
    size_t size;
    std::list<Shard> shards;
    TsfTaskWaiter* waiter;
    spdlog::stopwatch sw;
    TsfTask(const Type type, const Location location, const std::string& brief)
        : type{type}, location{location}, brief{brief}, id{0}, number{0}, size{0}, waiter{nullptr}, sw{}
    {
    }
    TsfTask() : TsfTask{Type::DUMP, Location::HOST, ""} {}
    Status Append(const std::string& blockId, const size_t offset, const uintptr_t address, const size_t length)
    {
        if (this->number == 0) { this->size = length; }
        if (this->size != length) { return Status::InvalidParam(); }
        this->shards.emplace_back<Shard>({this->number, blockId, offset, address});
        this->number++;
        return Status::OK();
    }
    std::string Str() { return fmt::format("{},{},{},{}", this->id, this->brief, this->size, this->number); }
};

} // namespace UC

#endif
