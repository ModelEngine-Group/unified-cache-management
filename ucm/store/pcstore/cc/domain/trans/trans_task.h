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
#include <chrono>
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
    static double NowTp() noexcept
    {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration<double>(now).count();
    }

public:
    enum class Type { DUMP, LOAD };
    size_t id;
    Type type;
    double startTp{0};
    static constexpr auto invalid = std::numeric_limits<size_t>::min();
    TransTask(Type&& type, std::string&& brief)
        : id{NextId()}, type{std::move(type)}, startTp{NowTp()}, brief_{std::move(brief)}
    {
    }
    void Append(const std::string& block, const uintptr_t address)
    {
        grouped_[block].push_back(address);
        number_++;
    }
    auto Str() const noexcept { return fmt::format("{},{},{}", id, brief_, number_); }
    size_t GroupNumber() const { return grouped_.size(); }
    size_t AddressNumber() const { return number_; }
    void ForEachGroup(std::function<void(const std::string&, std::vector<uintptr_t>&)> fn)
    {
        for (auto& [block, shards] : grouped_) { fn(block, shards); }
    }
    auto Epilog(const size_t ioSize) const noexcept
    {
        auto total = ioSize * number_;
        auto costs = NowTp() - startTp;
        auto bw = double(total) / costs / 1e9;
        return fmt::format("Task({},{},{},{}) finished, costs={:.06f}s, bw={:.06f}GB/s.", id,
                           brief_, number_, total, costs, bw);
    }

private:
    std::string brief_;
    size_t number_{0};
    std::unordered_map<std::string, std::vector<uintptr_t>> grouped_;
};

}  // namespace UC

#endif
