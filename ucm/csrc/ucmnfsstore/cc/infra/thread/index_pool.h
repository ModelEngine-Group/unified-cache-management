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
#ifndef UNIFIEDCACHE_INDEX_POOL_H
#define UNIFIEDCACHE_INDEX_POOL_H

#include <atomic>
#include <cstddef>
#include <limits>
#include <vector>

namespace UC {

class IndexPool {
public:
    using Index = uint32_t;
    static constexpr Index npos = std::numeric_limits<Index>::max();

private:
    struct Node {
        Index idx;
        Index next;
    };
    struct Pointer {
        Index slot;
        uint32_t ver;
    };
    static_assert(sizeof(Pointer) == 8, "Pointer must be 64-bit");

public:
    void Setup(const Index capacity) noexcept
    {
        this->_capacity = capacity;
        this->_nodes.resize(capacity + 1);
        for (Index slot = 1; slot <= capacity; slot++) {
            this->_nodes[slot].idx = slot - 1;
            this->_nodes[slot].next = slot + 1;
        }
        this->_nodes[capacity].next = 0;
        this->_pointer.store({1, 0});
    }
    Index Acquire() noexcept
    {
        for (;;) {
            auto ptr = this->_pointer.load(std::memory_order_acquire);
            if (ptr.slot == 0) { return npos; }
            auto next = this->_nodes[ptr.slot].next;
            Pointer desired{next, ptr.ver + 1};
            if (this->_pointer.compare_exchange_weak(ptr, desired, std::memory_order_release,
                                                     std::memory_order_relaxed)) {
                return this->_nodes[ptr.slot].idx;
            }
        }
    }
    void Release(const Index idx) noexcept
    {
        if (idx >= this->_capacity) { return; }
        auto slot = idx + 1;
        for (;;) {
            auto ptr = this->_pointer.load(std::memory_order_acquire);
            this->_nodes[slot].next = ptr.slot;
            Pointer desired{slot, ptr.ver + 1};
            if (this->_pointer.compare_exchange_weak(ptr, desired, std::memory_order_release,
                                                     std::memory_order_relaxed)) {
                return;
            }
        }
    }

private:
    Index _capacity;
    std::vector<Node> _nodes;
    alignas(64) std::atomic<Pointer> _pointer;
};

} // namespace UC

#endif
