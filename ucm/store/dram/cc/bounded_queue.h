/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_BOUNDED_QUEUE_H
#define UNIFIEDCACHE_DRAM_STORE_CC_BOUNDED_QUEUE_H

#include <cassert>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace UC::Dram {

// A fixed-capacity queue intended to be protected by its owner's mutex.
// Storage is allocated during composition/startup; Push/Pop perform no heap
// allocation and preserve the queue boundary's ownership semantics.
template <typename T>
class BoundedQueue final {
    static_assert(std::is_nothrow_move_constructible_v<T>,
                  "BoundedQueue payloads must transfer ownership without throwing");

public:
    explicit BoundedQueue(std::size_t capacity) : slots_(capacity) {}

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    bool Empty() const noexcept { return size_ == 0; }
    bool Full() const noexcept { return size_ == slots_.size(); }
    std::size_t Available() const noexcept { return slots_.size() - size_; }

    bool Push(T& value) noexcept
    {
        if (Full()) { return false; }
        assert(!slots_[tail_].has_value());
        slots_[tail_].emplace(std::move(value));
        tail_ = Next(tail_);
        ++size_;
        return true;
    }

    T Pop() noexcept
    {
        assert(!Empty());
        assert(slots_[head_].has_value());
        T value = std::move(*slots_[head_]);
        slots_[head_].reset();
        head_ = Next(head_);
        --size_;
        return value;
    }

private:
    std::size_t Next(std::size_t index) const noexcept
    {
        assert(!slots_.empty());
        ++index;
        return index == slots_.size() ? 0 : index;
    }

    std::vector<std::optional<T>> slots_;
    std::size_t head_{0};
    std::size_t tail_{0};
    std::size_t size_{0};
};

}  // namespace UC::Dram

#endif  // UNIFIEDCACHE_DRAM_STORE_CC_BOUNDED_QUEUE_H
