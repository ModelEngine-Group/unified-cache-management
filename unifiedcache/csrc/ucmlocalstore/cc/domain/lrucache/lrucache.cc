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
#include <unistd.h>
#include <algorithm>
#include <sys/mman.h>
#include <sys/fcntl.h>
#include "logger/logger.h"
#include "thread_pool/thread_pool.h"

#define UCM_LRU_CACHE_INVALID_32 0xFFFFFFFF
#define UCM_LRU_CACHE_INVALID_64 0xFFFFFFFFFFFFFFFF
#define UCM_LRU_CACHE_SHM_FILENAME "/ucm_lru_cache"
#define UCM_LRU_CACHE_INDEX_QUEUE_SHM_FILENAME "/ucm_lru_cache_index_queue"
#define UCM_LRU_CACHE_SHM_MAGIC 0x1122334455667788

namespace UCM {

struct alignas(1) Index {
    uint32_t idx;
    uint32_t version;

    Index() : idx{UCM_LRU_CACHE_INVALID_32}, version{0} {}
    Index(uint32_t idx, uint32_t version) : idx{idx}, version{version} {}
};
static_assert(sizeof(Index) == 8);

struct alignas(1) Key {
    uint64_t l;
    uint64_t r;

    Key() : l{UCM_LRU_CACHE_INVALID_64}, r{UCM_LRU_CACHE_INVALID_64} {}
    Key(std::string_view key)
    {
        std::memcpy(&(this->l), key.data(), 8);
        std::memcpy(&(this->r), key.data() + 8, 8);
    }

    bool operator==(const Key& other)
    {
        return (this->l == other.l && this->r == other.r);
    }
};
static_assert(sizeof(Key) == 16);

struct alignas(1) RefCnt {
    uint32_t val;
    uint32_t version;

    RefCnt() : val{0}, version{0} {}
    RefCnt(uint32_t val, uint32_t version) : val{val}, version{version} {}
};
static_assert(sizeof(RefCnt) == 8);

enum class State : uint64_t {
    INACTIVE = 0,
    WRITING,
    ACTIVE,
    READING,
    EVICTING,
};
static_assert(sizeof(State) == 8);

struct alignas(1) Info {
    std::atomic<Key> key;
    std::atomic<Index> prev;
    std::atomic<Index> next;
    std::atomic<State> state;
    std::atomic<RefCnt> ref_cnt;
    std::atomic<uint64_t> pinning_idx;
    std::atomic<uint64_t> access_time;

    void Initialize(uint64_t pinning_idx)
    {
        this->key.store(Key{}, std::memory_order_release);
        this->prev.store(Index{}, std::memory_order_release);
        this->next.store(Index{}, std::memory_order_release);
        this->state.store(State::INACTIVE, std::memory_order_release);
        this->ref_cnt.store(RefCnt{}, std::memory_order_release);
        this->pinning_idx.store(pinning_idx, std::memory_order_release);
        this->access_time.store(UCM_LRU_CACHE_INVALID_64, std::memory_order_release);
    }
};
static_assert(sizeof(Info) == 64);

struct alignas(1) Cache {
    Info info;
    std::byte data[];

    void Initialize(uint64_t pinning_idx)
    {
        this->info.Initialize(pinning_idx);
    }
};
static_assert(sizeof(Cache) == 64);

struct alignas(1) Length {
    uint32_t val;
    uint32_t version;

    Length() : val{0}, version{0} {}
    Length(uint32_t val, uint32_t version) : val{val}, version{version} {}
};
static_assert(sizeof(Length) == 8);

struct alignas(1) LRUCacheHeader {
    std::atomic<uint64_t> magic;
    std::atomic<uint64_t> cap;
    std::atomic<Length> len;
    std::atomic<Index> head;
    std::atomic<Index> tail;
    std::atomic<uint64_t> cache_size;
    std::atomic_bool removing;
    std::byte reserved[4047];
    Cache caches[];

    void Initialize(uint64_t cap, uint64_t cache_size)
    {
        this->cap.store(cap, std::memory_order_release);
        this->len.store(Length{}, std::memory_order_release);
        this->head.store(Index{}, std::memory_order_release);
        this->tail.store(Index{}, std::memory_order_release);
        this->cache_size.store(cache_size, std::memory_order_release);
        this->removing.store(false, std::memory_order_release);
        for (uint64_t i = 0; i < cap; ++i) {
            Cache* cache = this->At(i);
            cache->Initialize(i);
        }
        this->magic.store(UCM_LRU_CACHE_SHM_MAGIC, std::memory_order_release);
    }

    Cache* At(uint64_t i)
    {
        return reinterpret_cast<Cache*>(
            reinterpret_cast<std::byte*>(this->caches) + i * (static_cast<uint64_t>(sizeof(Info)) + cache_size)
        );
    }

    void Push(Cache* new_head_cache)
    {
        if (this->len.load(std::memory_order_acquire).val == this->cap.load(std::memory_order_relaxed)) {
            return;
        }

        while (this->len.load(std::memory_order_acquire).val == 0) {
            auto expected = Index{};
            auto ok = this->tail.compare_exchange_strong(
                expected,
                Index{static_cast<uint32_t>(new_head_cache->info.pinning_idx.load(std::memory_order_relaxed)), 1},
                std::memory_order_acq_rel
            );
            if (!ok) {
                continue;
            }
            this->head.store(
                Index{static_cast<uint32_t>(new_head_cache->info.pinning_idx.load(std::memory_order_relaxed)), 1},
                std::memory_order_release
            );
            this->len.store(Length{1, 1}, std::memory_order_release);
            return;
        }

        auto old_len = Length{};
        auto new_len = Length{};
        do {
            old_len = this->len.load(std::memory_order_acquire);
            new_len = Length{old_len.val + 1, old_len.version + 1};
        } while (!this->len.compare_exchange_strong(old_len, new_len, std::memory_order_acq_rel));

        auto old_head = Index{};
        auto new_head = Index{};
        do {
            old_head = this->head.load(std::memory_order_acquire);
            new_head = Index{
                static_cast<uint32_t>(new_head_cache->info.pinning_idx.load(std::memory_order_relaxed)),
                old_head.version + 1
            };
        } while (!this->head.compare_exchange_strong(old_head, new_head, std::memory_order_acq_rel));

        auto old_head_cache = this->At(old_head.idx);
        auto old_prev = old_head_cache->info.prev.load(std::memory_order_acquire);
        auto new_prev = Index{
            static_cast<uint32_t>(new_head_cache->info.pinning_idx.load(std::memory_order_relaxed)),
            old_prev.version + 1
        };
        old_head_cache->info.prev.store(new_prev, std::memory_order_release);

        auto old_next = new_head_cache->info.next.load(std::memory_order_acquire);
        auto new_next = Index{
            static_cast<uint32_t>(old_head_cache->info.pinning_idx.load(std::memory_order_relaxed)),
            old_next.version + 1
        };
        new_head_cache->info.next.store(new_next, std::memory_order_release);
    }

    void Pop()
    {
        if (this->len.load(std::memory_order_acquire).val != this->cap.load(std::memory_order_relaxed)) {
            return;
        }

        Cache* old_tail_cache = nullptr;
        Cache* new_tail_cache = nullptr;
        auto old_tail = Index{};
        auto new_tail = Index{};
        do {
            old_tail = this->tail.load(std::memory_order_release);
            old_tail_cache = this->At(old_tail.idx);
            new_tail_cache = this->At(old_tail_cache->info.prev.load(std::memory_order_acquire).idx);
            new_tail = Index{
                static_cast<uint32_t>(new_tail_cache->info.pinning_idx.load(std::memory_order_relaxed)),
                old_tail.version + 1
            };
        } while (!this->tail.compare_exchange_strong(old_tail, new_tail, std::memory_order_acq_rel));

        auto old_len = Length{};
        auto new_len = Length{};
        do {
            old_len = this->len.load(std::memory_order_acquire);
            new_len = Length{old_len.val - 1, old_len.version + 1};
        } while (!this->len.compare_exchange_strong(old_len, new_len, std::memory_order_acq_rel));
    }
};
static_assert(sizeof(LRUCacheHeader) == 4096);

LRUCache::LRUCache()
    : _h{nullptr}, _f{File::Make(UCM_LRU_CACHE_SHM_FILENAME)}, _q{UCM_LRU_CACHE_INDEX_QUEUE_SHM_FILENAME}
{}

LRUCache::~LRUCache()
{
    if (this->_h != nullptr) {
        File::MUnMap(
            reinterpret_cast<void*>(this->_h),
            static_cast<uint64_t>(sizeof(LRUCacheHeader)) +
            this->_h->cap.load(std::memory_order_relaxed) *
            (static_cast<uint64_t>(sizeof(Info)) + this->_h->cache_size.load(std::memory_order_relaxed))
        );
    }
}

Status LRUCache::MappingCheck(uint64_t shm_cap)
{
    auto status = Status::OK;

    status = this->_f->ShmOpen(OpenFlag::RDWR);
    if (status != Status::OK) {
        return status;
    }

    FileStat stat;
    do {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        status = this->_f->Stat(stat);
        if (status != Status::OK) {
            return status;
        }
    } while(static_cast<uint64_t>(stat.st_size) != shm_cap);

    status = this->_f->MMap(reinterpret_cast<void**>(&(this->_h)), shm_cap, PROT_READ | PROT_WRITE, MAP_SHARED);
    if (status != Status::OK) {
        return status;
    }

    while (this->_h->magic.load(std::memory_order_acquire) != UCM_LRU_CACHE_SHM_MAGIC) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    return status;
}

Status LRUCache::MappingInitialize(uint64_t shm_cap, uint64_t cache_num, uint64_t cache_size)
{
    auto status = this->_f->MMap(reinterpret_cast<void**>(&(this->_h)), shm_cap, PROT_READ | PROT_WRITE, MAP_SHARED);
    if (status != Status::OK) {
        return status;
    }
    this->_h->Initialize(cache_num, cache_size);
    return status;
}

Status LRUCache::Initialize(uint64_t cache_num, uint64_t cache_size)
{
    auto status = this->_q.Initialize(cache_num);
    if (status != Status::OK) {
        return status;
    }

    uint64_t shm_cap = sizeof(LRUCacheHeader) + cache_num * (sizeof(Info) + cache_size);

    status = this->_f->ShmOpen(OpenFlag::RDWR | OpenFlag::CREAT | OpenFlag::EXCL);
    if (status != Status::OK) {
        if (status == Status::EXIST) {
            return this->MappingCheck(shm_cap);
        } else {
            return status;
        }
    }

    status = this->_f->Truncate(shm_cap);
    if (status != Status::OK) {
        return status;
    }

    return this->MappingInitialize(shm_cap, cache_num, cache_size);
}

Status LRUCache::Insert(std::string_view key, void** val)
{
    *val = nullptr;
    if (key.size() != 16) { return Status::INVALID_PARAM; }

    if (this->_h->len.load(std::memory_order_acquire).val == this->_h->cap.load(std::memory_order_relaxed)) {
        ThreadPool::Instance().Submit([this] { this->Remove(); });
        return Status::BUSY;
    }

    uint64_t pinning_idx;
    auto status = this->_q.Pop(pinning_idx);
    if (status != Status::OK) {
        return status;
    }

    auto cache = this->_h->At(pinning_idx);

    auto expected = State::INACTIVE;
    if(!cache->info.state.compare_exchange_strong(expected, State::WRITING, std::memory_order_acq_rel)) {
        return Status::BUSY;
    }
    this->_h->Push(cache);
    cache->info.key.store(Key{key}, std::memory_order_release);
    *val = cache->data;

    return Status::BUSY;
}

Status LRUCache::Find(std::string_view key, void** val)
{
    *val = nullptr;
    if (key.size() != 16) { return Status::INVALID_PARAM; }

    if (this->_h->len.load(std::memory_order_acquire).val == 0) {
        return Status::EMPTY;
    }

    for (uint64_t i = 0; i < this->_h->cap.load(std::memory_order_relaxed); ++i) {
        auto cache = this->_h->At(i);

        if (cache->info.key.load(std::memory_order_acquire) == Key{key}) {
            while (true) {
                auto state = cache->info.state.load(std::memory_order_acquire);

                if (state == State::WRITING || state == State::EVICTING) {
                    return Status::BUSY;
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

                if (old_ref_cnt.val == 0) {
                    cache->info.state.store(State::READING, std::memory_order_release);
                }

                *val = cache->data;

                return Status::OK;
            }
        }
    }

    return Status::NOT_EXIST;
}

void LRUCache::Done(void* val)
{
    auto cache = reinterpret_cast<Cache*>(reinterpret_cast<std::byte*>(val) - sizeof(Info));

    auto state = cache->info.state.load(std::memory_order_acquire);
    if (state == State::WRITING) {
        cache->info.state.store(State::ACTIVE, std::memory_order_release);
    } else if (state == State::READING) {
        auto old_ref_cnt = RefCnt{};
        auto new_ref_cnt = RefCnt{};
        do {
            old_ref_cnt = cache->info.ref_cnt.load(std::memory_order_acquire);
            new_ref_cnt = RefCnt{old_ref_cnt.val - 1, old_ref_cnt.version + 1};
        } while (!cache->info.ref_cnt.compare_exchange_strong(old_ref_cnt, new_ref_cnt, std::memory_order_acq_rel));
        if (old_ref_cnt.val == 1) {
            cache->info.state.store(State::ACTIVE, std::memory_order_release);
        }
    } else {
        UCM_ERROR(
            "Failed to done, unexpected cache state = {}",
            static_cast<uint64_t>(cache->info.state.load(std::memory_order_acquire))
        );
    }
}

void LRUCache::Remove()
{
    auto expected = false;
    if (!this->_h->removing.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
    }

    Cache* tail_cache = nullptr;
    uint64_t cnt = 0;
    while (true) {
        if (cnt == this->_h->len.load(std::memory_order_acquire).val) {
            UCM_WARN("Failed to evict, no available cache.");
            expected = true;
            while (!this->_h->removing.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {}
            return;
        }
        ++cnt;

        tail_cache = this->_h->At(this->_h->tail.load(std::memory_order_acquire).idx);
        auto state = tail_cache->info.state.load(std::memory_order_acquire);
        if (state != State::READING || state != State::WRITING) {
            break;
        }

        this->_h->Pop();
        this->_h->Push(tail_cache);
    }

    auto expected_state = State::ACTIVE;
    if(!tail_cache->info.state.compare_exchange_strong(expected_state, State::EVICTING, std::memory_order_acq_rel)) {
        expected = true;
        while (!this->_h->removing.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {}
        return;
    }
    tail_cache->info.state.store(State::INACTIVE, std::memory_order_release);
    this->_h->Pop();
    this->_q.Push(tail_cache->info.pinning_idx.load(std::memory_order_relaxed));

    expected = true;
    while (!this->_h->removing.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {}
}

} // namespace UCM