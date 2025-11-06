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
#ifndef UNIFIEDCACHE_TRANS_TASK_H
#define UNIFIEDCACHE_TRANS_TASK_H

#include <atomic>
#include <fmt/format.h>
#include <functional>
#include <limits>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace UC {

class TransTask {
    static size_t NextId() noexcept
    {
        static std::atomic<size_t> id{invalid + 1};
        return id.fetch_add(1, std::memory_order_relaxed);
    };

public:
    enum class Type { DUMP, LOAD };
    struct Shard {
        uintptr_t address;
        size_t offset;
        size_t length;
        Shard(uintptr_t address, size_t offset, size_t length)
            : address{address}, offset{offset}, length{length}
        {
        }
    };
    Type type;
    static constexpr auto invalid = std::numeric_limits<size_t>::min();
    TransTask(Type&& type, std::string&& brief)
        : type{std::move(type)}, id_{NextId()}, brief_{std::move(brief)}
    {
    }
    void Append(const std::string& block, const size_t offset, const uintptr_t address,
                const size_t length)
    {
        auto& shard = shards_.emplace_back(address, offset, length);
        addresses_.push_back(address);
        grouped_[block].push_back(&shard);
        size_ += length;
    }
    size_t Id() const { return id_; }
    auto Str() const noexcept { return fmt::format("{},{},{},{}", id_, brief_, Number(), size_); }
    uintptr_t* Addresses() { return addresses_.data(); }
    size_t Number() const { return addresses_.size(); }
    size_t GroupNumber() const { return grouped_.size(); }
    void ForEachGroup(std::function<void(const std::string&, std::list<Shard*>&)> fn)
    {
        for (auto& [block, shards] : grouped_) { fn(block, shards); }
    }

private:
    size_t id_;
    std::string brief_;
    size_t size_{0};
    std::list<Shard> shards_;
    std::vector<uintptr_t> addresses_;
    std::unordered_map<std::string, std::list<Shard*>> grouped_;
};

} // namespace UC

#endif
