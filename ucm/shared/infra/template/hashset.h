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
#ifndef UNIFIEDCACHE_INFRA_HASHSET_H
#define UNIFIEDCACHE_INFRA_HASHSET_H

#include <array>
#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <vector>

namespace UC {

template <class Key, class Hash = std::hash<Key>, size_t ShardBits = 10>
class HashSet {
    static_assert(ShardBits <= 10, "ShardBits too large");
    static constexpr size_t Shards = size_t{1} << ShardBits;
    enum class SlotState : uint8_t { Empty, Occupied, Deleted };
    struct Slot {
        std::optional<Key> key;
        SlotState state = SlotState::Empty;
    };
    struct alignas(64) Shard {
        mutable std::shared_mutex mtx;
        std::vector<Slot> keys;
        size_t used = 0;
        size_t tombstones = 0;
    };
    std::array<Shard, Shards> shards_;
    Hash hash_;
    std::atomic<size_t> size_{0};
    static size_t ShardIndex(size_t h) noexcept { return h & (Shards - 1); }
    static size_t Probe(size_t idx, size_t cap) noexcept { return (idx + 1) & (cap - 1); }
    static bool IsEmpty(const Slot& slot) noexcept { return slot.state == SlotState::Empty; }
    static bool IsOccupied(const Slot& slot) noexcept { return slot.state == SlotState::Occupied; }
    static bool IsDeleted(const Slot& slot) noexcept { return slot.state == SlotState::Deleted; }
    void RehashShard(Shard& s)
    {
        std::vector<Slot> old = std::move(s.keys);
        size_t new_cap = (old.empty() ? 8 : old.size() * 2);
        s.keys.assign(new_cap, Slot{});
        s.used = 0;
        s.tombstones = 0;
        for (const auto& slot : old) {
            if (!IsOccupied(slot)) { continue; }
            const Key& k = *slot.key;
            size_t h = hash_(k);
            size_t idx = (h >> ShardBits) & (new_cap - 1);
            while (!IsEmpty(s.keys[idx])) { idx = Probe(idx, new_cap); }
            s.keys[idx].key.emplace(k);
            s.keys[idx].state = SlotState::Occupied;
            ++s.used;
        }
    }

public:
    void Insert(const Key& key)
    {
        size_t h = hash_(key);
        auto& s = shards_[ShardIndex(h)];
        std::unique_lock lg(s.mtx);
        if ((s.used + s.tombstones) * 4 >= s.keys.size() * 3) [[unlikely]] { RehashShard(s); }
        size_t cap = s.keys.size();
        if (cap == 0) {
            RehashShard(s);
            cap = s.keys.size();
        }
        size_t idx = (h >> ShardBits) & (cap - 1);
        size_t start = idx;
        size_t first_deleted = cap;
        do {
            if (IsOccupied(s.keys[idx]) && *s.keys[idx].key == key) { return; }
            if (IsDeleted(s.keys[idx]) && first_deleted == cap) { first_deleted = idx; }
            if (IsEmpty(s.keys[idx])) {
                size_t target = (first_deleted != cap) ? first_deleted : idx;
                auto& slot = s.keys[target];
                if (IsDeleted(slot)) { --s.tombstones; }
                slot.key.emplace(key);
                slot.state = SlotState::Occupied;
                ++s.used;
                ++size_;
                return;
            }
            idx = Probe(idx, cap);
        } while (idx != start);
        if (first_deleted != cap) {
            auto& slot = s.keys[first_deleted];
            --s.tombstones;
            slot.key.emplace(key);
            slot.state = SlotState::Occupied;
            ++s.used;
            ++size_;
            return;
        }
        RehashShard(s);
        cap = s.keys.size();
        idx = (h >> ShardBits) & (cap - 1);
        while (!IsEmpty(s.keys[idx])) { idx = Probe(idx, cap); }
        s.keys[idx].key.emplace(key);
        s.keys[idx].state = SlotState::Occupied;
        ++s.used;
        ++size_;
    }
    bool Contains(const Key& key) const
    {
        size_t h = hash_(key);
        auto& s = shards_[ShardIndex(h)];
        std::shared_lock lg(s.mtx);
        size_t cap = s.keys.size();
        if (cap == 0) { return false; }
        size_t idx = (h >> ShardBits) & (cap - 1);
        size_t start = idx;
        do {
            if (IsEmpty(s.keys[idx])) { break; }
            if (IsOccupied(s.keys[idx]) && *s.keys[idx].key == key) { return true; }
            idx = Probe(idx, cap);
        } while (idx != start);
        return false;
    }
    void Remove(const Key& key)
    {
        size_t h = hash_(key);
        auto& s = shards_[ShardIndex(h)];
        std::unique_lock lg(s.mtx);
        size_t cap = s.keys.size();
        if (cap == 0) { return; }
        size_t idx = (h >> ShardBits) & (cap - 1);
        size_t start = idx;
        do {
            if (IsEmpty(s.keys[idx])) { break; }
            if (IsOccupied(s.keys[idx]) && *s.keys[idx].key == key) {
                s.keys[idx].key.reset();
                s.keys[idx].state = SlotState::Deleted;
                --s.used;
                ++s.tombstones;
                --size_;
                return;
            }
            idx = Probe(idx, cap);
        } while (idx != start);
    }
};

} // namespace UC

#endif
