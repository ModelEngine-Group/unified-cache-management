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
#include "cache_index.h"
#include <algorithm>
#include <atomic>

namespace UC {

struct Node {
    uint32_t idx;
    uint32_t next;
};
struct Pointer {
    uint32_t slot;
    uint32_t ver;
};
struct Header {
    uint32_t magic;
    uint32_t capacity;
    alignas(64) std::atomic<Pointer> pointer;
    uint64_t padding[7];
    Node nodes[0];
};
static_assert(sizeof(Pointer) == 8, "Pointer must be 64-bit");
static_assert(sizeof(Header) == 128, "Header must be 128-Byte");

static constexpr uint32_t Magic = (('C' << 16) | ('i' << 8) | 1);
inline auto HeaderPtr(void* addr) { return (Header*)addr; }

size_t CacheIndex::MemorySize() const noexcept
{
    return sizeof(Header) + sizeof(Node) * (this->capacity_ + 1);
}

void CacheIndex::Setup(void* addr) noexcept
{
    this->addr_ = addr;
    auto header = HeaderPtr(this->addr_);
    if (header->magic == Magic) { return; }
    header->capacity = this->capacity_;
    header->pointer.store({1, 0});
    auto paddingSize = sizeof(header->padding) / sizeof(*header->padding);
    std::fill_n(header->padding, paddingSize, 0);
    for (uint32_t slot = 1; slot <= header->capacity; slot++) {
        header->nodes[slot].idx = slot - 1;
        header->nodes[slot].next = slot + 1;
    }
    header->nodes[header->capacity].next = 0;
    header->magic = Magic;
    return;
}

uint32_t CacheIndex::Acquire() noexcept
{
    auto header = HeaderPtr(this->addr_);
    for (;;) {
        auto ptr = header->pointer.load(std::memory_order_acquire);
        if (ptr.slot == 0) { return npos; }
        auto next = header->nodes[ptr.slot].next;
        Pointer desired{next, ptr.ver + 1};
        if (header->pointer.compare_exchange_weak(ptr, desired, std::memory_order_release,
                                                  std::memory_order_relaxed)) {
            return header->nodes[ptr.slot].idx;
        }
    }
}

void CacheIndex::Release(const uint32_t idx) noexcept
{
    auto header = HeaderPtr(this->addr_);
    if (idx >= header->capacity) { return; }
    auto slot = idx + 1;
    for (;;) {
        auto ptr = header->pointer.load(std::memory_order_acquire);
        header->nodes[slot].next = ptr.slot;
        Pointer desired{slot, ptr.ver + 1};
        if (header->pointer.compare_exchange_weak(ptr, desired, std::memory_order_release,
                                                  std::memory_order_relaxed)) {
            return;
        }
    }
}

} // namespace UC
