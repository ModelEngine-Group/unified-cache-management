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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_BLOCK_TRACKER_H
#define UNIFIEDCACHE_POSIX_STORE_CC_BLOCK_TRACKER_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>
#include "type/types.h"

namespace UC::PosixStore {

inline uint64_t InoOf(const std::string& path)
{
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) { return 0; }
    return static_cast<uint64_t>(st.st_ino);
}

inline uint64_t InoOf(int32_t fd)
{
    struct stat st{};
    if (::fstat(fd, &st) != 0) { return 0; }
    return static_cast<uint64_t>(st.st_ino);
}

class BlockCommitTracker {
public:
    static constexpr size_t kShardBits = 10;
    static constexpr size_t kShards = size_t{1} << kShardBits;

    BlockCommitTracker() = default;
    BlockCommitTracker(const BlockCommitTracker&) = delete;
    BlockCommitTracker& operator=(const BlockCommitTracker&) = delete;

    void Setup(size_t nShardPerBlock)
    {
        nShardPerBlock_ = nShardPerBlock;
        nWords_ = (nShardPerBlock + 63) / 64;
        fullMask_.assign(nWords_, 0);
        for (size_t i = 0; i < nWords_; ++i) {
            const auto bits = std::min<size_t>(64, nShardPerBlock - i * 64);
            fullMask_[i] = bits == 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1;
        }
    }

    bool MarkDone(const Detail::BlockId& id, size_t index, uint64_t ino)
    {
        if (index >= nShardPerBlock_) [[unlikely]] { return false; }
        auto& bucket = BucketOf(id);
        std::lock_guard<std::mutex> lock{bucket.mtx};
        auto it = bucket.map.find(Key{id, ino});
        if (it == bucket.map.end()) { it = bucket.map.emplace(Key{id, ino}, Entry{nWords_}).first; }
        it->second.bits[index / 64] |= uint64_t{1} << (index % 64);
        if (!Complete(it->second)) { return false; }
        bucket.map.erase(it);
        return true;
    }

private:
    struct Key {
        Detail::BlockId id;
        uint64_t ino{0};
        bool operator==(const Key& other) const noexcept
        {
            return ino == other.ino && id == other.id;
        }
    };
    struct KeyHasher {
        size_t operator()(const Key& key) const noexcept
        {
            Detail::BlockIdHasher hasher;
            const auto h1 = hasher(key.id);
            const auto h2 = std::hash<uint64_t>{}(key.ino);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };
    struct Entry {
        std::vector<uint64_t> bits;
        explicit Entry(size_t words) : bits(words, 0) {}
    };
    struct alignas(64) Bucket {
        std::mutex mtx;
        std::unordered_map<Key, Entry, KeyHasher> map;
    };

    bool Complete(const Entry& entry) const
    {
        for (size_t i = 0; i < nWords_; ++i) {
            if (entry.bits[i] != fullMask_[i]) { return false; }
        }
        return true;
    }

    Bucket& BucketOf(const Detail::BlockId& id)
    {
        Detail::BlockIdHasher hasher;
        return buckets_[hasher(id) & (kShards - 1)];
    }

    size_t nShardPerBlock_{0};
    size_t nWords_{0};
    std::vector<uint64_t> fullMask_{};
    std::array<Bucket, kShards> buckets_{};
};

}  // namespace UC::PosixStore

#endif
