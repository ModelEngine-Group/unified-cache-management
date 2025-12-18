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
#ifndef UNIFIEDCACHE_TASK_SHARD_H
#define UNIFIEDCACHE_TASK_SHARD_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <fmt/format.h>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <vector>
#include "logger/logger.h"
#include "task_waiter.h"
#include "metrics_api.h"

namespace UC {

class Task {
public:
    enum class Type { DUMP, LOAD };
    enum class Location { HOST, DEVICE };
    struct Shard {
        Type type;
        Location location;
        std::string block;
        size_t offset;
        uintptr_t address;
        size_t length;
        size_t owner;
        std::shared_ptr<void> buffer;
        std::function<void(void)> done;
        Shard(const Type type, const Location location, const std::string& block,
              const size_t offset, const uintptr_t address, const size_t length, const size_t owner)
            : type{type}, location{location}, block{block}, offset{offset}, address{address},
              length{length}, owner{owner}, buffer{nullptr}, done{nullptr}
        {
        }
        Shard(const Shard&) = delete;
        Shard& operator=(const Shard&) = delete;
        Shard& operator=(Shard&& s) noexcept
        {
            if (this != &s) {
                this->type = s.type;
                this->location = s.location;
                this->block = std::move(s.block);
                this->offset = s.offset;
                this->address = s.address;
                this->length = s.length;
                this->owner = s.owner;
                this->buffer = std::move(s.buffer);
                this->done = std::move(s.done);
            }
            return *this;
        }
        Shard(Shard&& s) noexcept { *this = std::move(s); }
    };
    static constexpr auto invalid = std::numeric_limits<size_t>::min();
    Task(Type&& type, Location&& location, std::string&& brief)
        : id_{NextId()}, type_{type}, location_{location}, brief_{std::move(brief)}, number_{0},
          size_{0}, startTp_{NowTp()}, execTp_{0.f}
    {
    }
    auto Id() const noexcept { return id_; }
    auto StartTp() const noexcept { return startTp_; }
    auto Str() const noexcept { return fmt::format("{},{},{},{}", id_, brief_, number_, size_); }
    void Append(const std::string& block, const size_t offset, const uintptr_t address,
                const size_t length)
    {
        shards_.emplace_back(type_, location_, block, offset, address, length, id_);
        number_++;
        size_ += length;
    }
    std::vector<std::list<Shard>> Split(const size_t n, std::shared_ptr<TaskWaiter> waiter)
    {
        auto num = std::min(n, number_);
        std::vector<std::list<Shard>> out(num);
        waiter->Set(num);
        auto base = number_ / num;
        auto rem = number_ % num;
        auto it = shards_.cbegin();
        for (size_t i = 0; i < num; i++) {
            auto next = std::next(it, base + (i < rem ? 1 : 0));
            out[i].splice(out[i].end(), shards_, it, next);
            out[i].back().done = [waiter, this] {
                waiter->Done([this] { UC_DEBUG("Task({}) finished, {}.", Str(), Stat()); });
            };
            it = next;
        }
        this->execTp_ = NowTp();
        return out;
    }

private:
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
    std::string Stat() const noexcept
    {
        auto wait = execTp_ - startTp_;
        auto exec = NowTp() - execTp_;
        auto bw = size_ / exec / 1024 / 1024 / 1024;
        switch (type_)
        {
        case Type::DUMP:
            UC::Metrics::UpdateStats("d2s_bandwidth", bw);
            break;
        case Type::LOAD:
            UC::Metrics::UpdateStats("s2d_bandwidth", bw);
            break;
        default:
            break;
        }
        return fmt::format("wait={:.06f}s, exec={:.06f}s, bw={:.06f}GB/s", wait, exec, bw);
    }

private:
    size_t id_;
    Type type_;
    Location location_;
    std::string brief_;
    std::list<Shard> shards_;
    size_t number_;
    size_t size_;
    double startTp_;
    double execTp_;
};

} // namespace UC

#endif
