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
#ifndef UNIFIEDCACHE_HASHMAP_H
#define UNIFIEDCACHE_HASHMAP_H

#include <array>
#include <atomic>
#include <functional>
#include <optional>
#include <shared_mutex>
#include <vector>

namespace UC {

template <class Key, class Value, class Hash = std::hash<Key>, size_t ShardBits = 10>
class HashMap {
    static_assert(ShardBits <= 10, "ShardBits too large");
    static constexpr size_t Shards = size_t{1} << ShardBits;

    struct alignas(64) Shard {
        mutable std::shared_mutex mtx;
        std::vector<std::pair<std::optional<Key>, std::optional<Value>>> slots;
        size_t used = 0;
    };

    std::array<Shard, Shards> shards_;
    Hash hash_;
    std::atomic<size_t> size_{0};

    static size_t ShardIndex(size_t h) noexcept {
        return h & (Shards - 1);
    }

    static size_t ProbeIdx(size_t idx, size_t cap) noexcept {
        return (idx + 1) & (cap - 1);
    }

    static bool IsEmpty(const std::optional<Key>& slot) noexcept {
        return !slot.has_value();
    }

    void RehashShard(Shard& s) {
        std::vector<std::pair<std::optional<Key>, std::optional<Value>>> old =
            std::move(s.slots);
        size_t new_cap = (old.empty() ? 8 : old.size() * 2);
        s.slots.assign(new_cap, {std::optional<Key>{}, std::optional<Value>{}});
        s.used = 0;

        for (const auto& slot : old) {
            if (!slot.first.has_value()) {
                continue;
            }

            const Key& k = *slot.first;
            const Value& v = *slot.second;
            size_t h = hash_(k);
            size_t idx = (h >> ShardBits) & (new_cap - 1);

            while (!IsEmpty(s.slots[idx].first)) {
                idx = ProbeIdx(idx, new_cap);
            }

            s.slots[idx].first.emplace(k);
            s.slots[idx].second.emplace(v);
            ++s.used;
        }
    }

public:
    HashMap() = default;
    std::optional<std::reference_wrapper<Value>> GetOrCreate(
        const Key& key,
        std::function<bool(Value&)> creator)
    {
        size_t h = hash_(key);
        auto& shard = shards_[ShardIndex(h)];
        std::unique_lock lg(shard.mtx);

        if (shard.used * 4 >= shard.slots.size() * 3) [[unlikely]] {
            RehashShard(shard);
        }

        size_t cap = shard.slots.size();
        if (cap == 0) {
            RehashShard(shard);
            cap = shard.slots.size();
        }

        size_t idx = (h >> ShardBits) & (cap - 1);
        size_t start = idx;

        do {
            if (shard.slots[idx].first.has_value() && *shard.slots[idx].first == key) {
                return std::ref(*shard.slots[idx].second);
            }
            if (IsEmpty(shard.slots[idx].first)) {
                Value newValue;
                if (!creator(newValue)) {
                    return std::optional<std::reference_wrapper<Value>>{};
                }
                shard.slots[idx].first.emplace(key);
                shard.slots[idx].second.emplace(std::move(newValue));
                ++shard.used;
                ++size_;
                return std::ref(*shard.slots[idx].second);
            }
            idx = ProbeIdx(idx, cap);
        } while (idx != start);
        RehashShard(shard);
        return GetOrCreate(key, creator);
    }

    void Upsert(const Key& key, std::function<bool(Value&)> updater) {
        size_t h = hash_(key);
        auto& shard = shards_[ShardIndex(h)];
        std::unique_lock lg(shard.mtx);

        size_t cap = shard.slots.size();
        if (cap == 0) {
            return;
        }

        size_t idx = (h >> ShardBits) & (cap - 1);
        size_t start = idx;

        do {
            if (shard.slots[idx].first.has_value() && *shard.slots[idx].first == key) {
                bool shouldDelete = updater(*shard.slots[idx].second);
                if (shouldDelete) {
                    shard.slots[idx].first.reset();
                    shard.slots[idx].second.reset();
                    --shard.used;
                    --size_;
                }
                return;
            }

            if (IsEmpty(shard.slots[idx].first)) {
                return;
            }

            idx = ProbeIdx(idx, cap);
        } while (idx != start);
    }

    void ForEach(std::function<void(const Key&, Value&)> visitor) {
        for (auto& shard : shards_) {
            std::shared_lock lg(shard.mtx);
            for (auto& slot : shard.slots) {
                if (slot.first.has_value()) {
                    visitor(*slot.first, *slot.second);
                }
            }
        }
    }

    void Clear() {
        for (auto& shard : shards_) {
            std::unique_lock lg(shard.mtx);
            shard.slots.clear();
            shard.used = 0;
        }
        size_.store(0);
    }
};

} // namespace UC

#endif
