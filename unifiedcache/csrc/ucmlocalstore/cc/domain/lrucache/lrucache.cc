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
#include "lrucache.h"
#include <algorithm>
#include <sys/mman.h>
#include "logger/logger.h"

#define UCM_LRU_CACHE_INVALID_32 0xFFFFFFFF
#define UCM_LRU_CACHE_INVALID_64 0xFFFFFFFFFFFFFFFF
#define UCM_LRU_CACHE_SHM_FILENAME "/ucm_lru_cache"
#define UCM_LRU_CACHE_INDEX_QUEUE_SHM_FILENAME "/ucm_lru_cache_index_queue"
#define UCM_LRU_CACHE_HASHMAP_SHM_FILENAME "/ucm_lru_cache_hashmap"
#define UCM_LRU_CACHE_SHM_MAGIC 0x89ABCDEF

namespace UCM {

struct Index {
    uint32_t val;
    uint32_t version;

    Index() : val{UCM_LRU_CACHE_INVALID_32}, version{0} {}
    Index(uint32_t val, uint32_t version) : val{val}, version{version} {}
};

struct RefCnt {
    uint32_t val;
    uint32_t version;

    RefCnt() : val{0}, version{0} {}
    RefCnt(uint32_t val, uint32_t version) : val{val}, version{version} {}

    bool operator==(const uint32_t val) const {
        return this->val == val;
    }

    bool operator!=(const uint32_t val) const {
        return this->val != val;
    }
};

struct Key {
    char data[16];

    Key() { std::fill(std::begin(this->data), std::end(this->data), static_cast<char>(0)); }
    Key(std::string_view key) { std::copy(key.begin(), key.end(), std::begin(this->data)); }

    std::string_view Unwrap() const { return {this->data, 16}; }
};


enum class State : uint32_t {
    INACTIVE = 0,
    WRITING,
    ACTIVE,
    READING,
    EVICTING,
};

struct Info {
    std::atomic<Key> key;
    std::atomic<Index> prev;
    std::atomic<Index> next;
    std::atomic<RefCnt> ref_cnt;
    std::atomic<uint32_t> pinning_idx;
    std::atomic<State> state;
    std::atomic<uint64_t> access_time;

    void Initialize(uint32_t pinning_idx)
    {
        this->key.store(Key{}, std::memory_order_release);
        this->prev.store(Index{}, std::memory_order_release);
        this->next.store(Index{}, std::memory_order_release);
        this->ref_cnt.store(RefCnt{}, std::memory_order_release);
        this->pinning_idx.store(pinning_idx, std::memory_order_release);
        this->state.store(State::INACTIVE, std::memory_order_release);
        this->access_time.store(UCM_LRU_CACHE_INVALID_64, std::memory_order_release);
    }

    void Reset()
    {
        this->key.store(Key{}, std::memory_order_release);
        this->prev.store(Index{}, std::memory_order_release);
        this->next.store(Index{}, std::memory_order_release);
        this->ref_cnt.store(RefCnt{}, std::memory_order_release);
        this->state.store(State::INACTIVE, std::memory_order_release);
        this->access_time.store(UCM_LRU_CACHE_INVALID_64, std::memory_order_release);
    }
};

struct Cache {
    Info info;
    std::byte data[];

    void Initialize(uint32_t pinning_idx)
    {
        this->info.Initialize(pinning_idx);
    }
};

struct Length {
    uint32_t val;
    uint32_t version;

    Length() : val{0}, version{0} {}
    Length(uint32_t val, uint32_t version) : val{val}, version{version} {}

    bool operator==(const uint32_t val) const {
        return this->val == val;
    }
};

struct LRUCacheHeader {
    std::atomic<uint32_t> magic;
    std::atomic<uint32_t> cap;
    std::atomic<Length> len;
    std::atomic<Index> head;
    std::atomic<Index> tail;
    std::atomic<uint32_t> cache_size;
    std::atomic_bool removing;
    std::byte padding[3];
    std::byte reserved[4056];
    Cache caches[];

    void Initialize(uint32_t cap, uint32_t cache_size)
    {
        this->cap.store(cap, std::memory_order_release);
        this->len.store(Length{}, std::memory_order_release);
        this->head.store(Index{}, std::memory_order_release);
        this->tail.store(Index{}, std::memory_order_release);
        this->cache_size.store(cache_size, std::memory_order_release);
        this->removing.store(false, std::memory_order_release);
        for (uint32_t i = 0; i < cap; ++i) {
            this->At(i)->Initialize(i);
        }
        this->magic.store(UCM_LRU_CACHE_SHM_MAGIC, std::memory_order_release);
    }

    Cache* At(uint32_t i)
    {
        return reinterpret_cast<Cache*>(
            reinterpret_cast<std::byte*>(this->caches) +
            static_cast<uint64_t>(i) *
            static_cast<uint64_t>(sizeof(Info) + this->cache_size.load(std::memory_order_relaxed))
        );
    }

    void Push(Cache* new_head_cache)
    {
        while (this->len.load(std::memory_order_acquire) == 0) {
            auto expected = Index{};
            if(
                !this->tail.compare_exchange_strong(
                    expected,
                    Index{new_head_cache->info.pinning_idx.load(std::memory_order_relaxed), 1},
                    std::memory_order_acq_rel
                )
            ) {
                continue;
            }
            this->head.store(
                Index{new_head_cache->info.pinning_idx.load(std::memory_order_relaxed), 1},
                std::memory_order_release
            );
            this->len.store(Length{1, 1}, std::memory_order_release);
            return;
        }

        auto old_len = Length{};
        auto new_len = Length{};
        do {
            old_len = this->len.load(std::memory_order_acquire);
            if (old_len == this->cap.load(std::memory_order_relaxed)) { return; }
            new_len = Length{old_len.val + 1, old_len.version + 1};
        } while (!this->len.compare_exchange_strong(old_len, new_len, std::memory_order_acq_rel));

        auto old_head = Index{};
        auto new_head = Index{};
        do {
            old_head = this->head.load(std::memory_order_acquire);
            new_head = Index{
                new_head_cache->info.pinning_idx.load(std::memory_order_relaxed),
                old_head.version + 1
            };
        } while (!this->head.compare_exchange_strong(old_head, new_head, std::memory_order_acq_rel));

        auto old_head_cache = this->At(old_head.val);
        auto old_prev = old_head_cache->info.prev.load(std::memory_order_acquire);
        auto new_prev = Index{
            new_head_cache->info.pinning_idx.load(std::memory_order_relaxed),
            old_prev.version + 1
        };
        old_head_cache->info.prev.store(new_prev, std::memory_order_release);

        auto old_next = new_head_cache->info.next.load(std::memory_order_acquire);
        auto new_next = Index{
            old_head_cache->info.pinning_idx.load(std::memory_order_relaxed),
            old_next.version + 1
        };
        new_head_cache->info.next.store(new_next, std::memory_order_release);
    }

    void Pop()
    {
        auto old_len = Length{};
        auto new_len = Length{};
        do {
            old_len = this->len.load(std::memory_order_acquire);
            if (old_len == 0) { return; }
            new_len = Length{old_len.val - 1, old_len.version + 1};
        } while (!this->len.compare_exchange_strong(old_len, new_len, std::memory_order_acq_rel));

        Cache* old_tail_cache = nullptr;
        Cache* new_tail_cache = nullptr;
        auto old_tail = Index{};
        auto new_tail = Index{};
        do {
            old_tail = this->tail.load(std::memory_order_release);
            old_tail_cache = this->At(old_tail.val);
            new_tail_cache = this->At(old_tail_cache->info.prev.load(std::memory_order_acquire).val);
            new_tail = Index{
                new_tail_cache->info.pinning_idx.load(std::memory_order_relaxed),
                old_tail.version + 1
            };
        } while (!this->tail.compare_exchange_strong(old_tail, new_tail, std::memory_order_acq_rel));
    }
};
static_assert(sizeof(LRUCacheHeader) == 4096);

LRUCache::LRUCache()
    : _h{nullptr}, _f{File::Make(UCM_LRU_CACHE_SHM_FILENAME)},
      _q{UCM_LRU_CACHE_INDEX_QUEUE_SHM_FILENAME}, _m{UCM_LRU_CACHE_HASHMAP_SHM_FILENAME}
{}

LRUCache::~LRUCache()
{
    if (this->_h != nullptr) {
        File::MUnMap(
            reinterpret_cast<void*>(this->_h),
            static_cast<uint64_t>(sizeof(LRUCacheHeader)) +
            static_cast<uint64_t>(this->_h->cap.load(std::memory_order_relaxed)) *
            static_cast<uint64_t>(sizeof(Info) + this->_h->cache_size.load(std::memory_order_relaxed))
        );
    }
}

Status LRUCache::Initialize(const uint32_t cache_num, const uint32_t cache_size)
{
    auto s = this->_q.Initialize(cache_num);
    if (s.Failure()) {
        return s;
    }

    s = this->_m.Initialize(cache_num);
    if (s.Failure()) {
        return s;
    }

    uint64_t shm_size = static_cast<uint64_t>(sizeof(LRUCacheHeader)) +
                        static_cast<uint64_t>(cache_num) *
                        static_cast<uint64_t>(sizeof(Info) + cache_size);

    s = this->_f->ShmOpen(OpenFlag::RDWR | OpenFlag::CREAT | OpenFlag::EXCL);
    if (s.Failure()) {
        if (s == Status::Exist()) {
            return this->MappingCheck(shm_size);
        } else {
            return s;
        }
    }

    s = this->_f->Truncate(shm_size);
    if (s.Failure()) {
        return s;
    }

    return this->MappingInitialize(shm_size, cache_num, cache_size);
}

Status LRUCache::Insert(std::string_view key, void*& val)
{
    val = nullptr;
    if (key.size() != 16) { return Status::InvalidParam(); }

    auto pinning_idx = this->_q.Pop();
    if (!pinning_idx.has_value()) {
        if (this->_h->len.load(std::memory_order_acquire) == this->_h->cap.load(std::memory_order_relaxed)) {
            std::thread([this] { this->Remove(); }).detach();
        }
        return Status::Busy();
    }
    auto s = this->_m.Insert(key, *pinning_idx);
    if (s.Failure()) {
        this->_q.Push(*pinning_idx);
        return s;
    }

    auto cache = this->_h->At(*pinning_idx);

    cache->info.state.store(State::WRITING, std::memory_order_release);
    cache->info.key.store(Key{key}, std::memory_order_release);

    this->_h->Push(cache);

    val = cache->data;

    return s;
}

Status LRUCache::Find(std::string_view key, void*& val)
{
    val = nullptr;
    if (key.size() != 16) { return Status::InvalidParam(); }

    auto pinning_idx = this->_m.Find(key);
    if (!pinning_idx.has_value()) {
        return Status::NotFound();
    }

    auto cache = this->_h->At(*pinning_idx);

    while (true) {
        auto state = cache->info.state.load(std::memory_order_acquire);

        if (state == State::WRITING) {
            return Status::Busy();
        }

        if (state == State::INACTIVE || state == State::EVICTING) {
            return Status::NotFound();
        }

        if (state == State::ACTIVE) {
            if(!cache->info.state.compare_exchange_strong(state, State::READING, std::memory_order_acq_rel)) {
                continue;
            }
        }

        auto old_ref_cnt = RefCnt{};
        auto new_ref_cnt = RefCnt{};
        do {
            old_ref_cnt = cache->info.ref_cnt.load(std::memory_order_acquire);
            new_ref_cnt = RefCnt{old_ref_cnt.val + 1, old_ref_cnt.version + 1};
        } while (!cache->info.ref_cnt.compare_exchange_strong(old_ref_cnt, new_ref_cnt, std::memory_order_acq_rel));

        if (cache->info.ref_cnt.load(std::memory_order_acquire) == 1) {
            cache->info.state.store(State::READING, std::memory_order_release);
        }

        val = cache->data;

        break;
    }

    return Status::OK();
}

void LRUCache::Done(void* val)
{
    auto cache = reinterpret_cast<Cache*>(reinterpret_cast<std::byte*>(val) - sizeof(Info));

    auto state = cache->info.state.load(std::memory_order_acquire);
    if (state == State::WRITING) {
        cache->info.state.store(State::ACTIVE, std::memory_order_release);
    } else {
        auto old_ref_cnt = RefCnt{};
        auto new_ref_cnt = RefCnt{};
        do {
            old_ref_cnt = cache->info.ref_cnt.load(std::memory_order_acquire);
            new_ref_cnt = RefCnt{old_ref_cnt.val - 1, old_ref_cnt.version + 1};
        } while (!cache->info.ref_cnt.compare_exchange_strong(old_ref_cnt, new_ref_cnt, std::memory_order_acq_rel));

        if (cache->info.ref_cnt.load(std::memory_order_acquire) == 0) {
            cache->info.state.store(State::ACTIVE, std::memory_order_release);
        }
    }
}

void LRUCache::Remove()
{
    auto expected = false;
    if (!this->_h->removing.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
    }

    Cache* tail_cache = nullptr;
    uint32_t cnt = 0;
    while (true) {
        if (this->_h->len.load(std::memory_order_acquire) == cnt) {
            UCM_WARN("Failed to evict, no available cache.");
            expected = true;
            while (!this->_h->removing.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {}
            return;
        }
        ++cnt;

        tail_cache = this->_h->At(this->_h->tail.load(std::memory_order_acquire).val);

        auto state = tail_cache->info.state.load(std::memory_order_acquire);
        if (state != State::READING && state != State::WRITING) {
            auto expected_state = State::ACTIVE;
            if(!tail_cache->info.state.compare_exchange_strong(expected_state, State::EVICTING, std::memory_order_acq_rel)) {
                --cnt;
                continue;
            } else {
                break;
            }
        }

        this->_h->Pop();
        this->_h->Push(tail_cache);
    }

    this->_m.Remove(tail_cache->info.key.load(std::memory_order_acquire).Unwrap());
    this->_h->Pop();
    tail_cache->info.state.store(State::INACTIVE, std::memory_order_release);
    this->_q.Push(tail_cache->info.pinning_idx.load(std::memory_order_relaxed));

    expected = true;
    while (!this->_h->removing.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {}
}

Status LRUCache::MappingCheck(const uint64_t shm_size)
{
    auto s = Status::OK();

    s = this->_f->ShmOpen(OpenFlag::RDWR);
    if (s.Failure()) {
        return s;
    }

    FileStat stat;
    do {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        s = this->_f->Stat(stat);
        if (s.Failure()) {
            return s;
        }
    } while(static_cast<uint64_t>(stat.st_size) != shm_size);

    s = this->_f->MMap(reinterpret_cast<void**>(&(this->_h)), shm_size, PROT_READ | PROT_WRITE, MAP_SHARED);
    if (s.Failure()) {
        return s;
    }

    while (this->_h->magic.load(std::memory_order_acquire) != UCM_LRU_CACHE_SHM_MAGIC) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    return s;
}

Status LRUCache::MappingInitialize(const uint64_t shm_size, const uint32_t cache_num, const uint32_t cache_size)
{
    auto s = this->_f->MMap(reinterpret_cast<void**>(&(this->_h)), shm_size, PROT_READ | PROT_WRITE, MAP_SHARED);
    if (s.Failure()) {
        return s;
    }
    this->_h->Initialize(cache_num, cache_size);
    return s;
}

} // namespace UCM