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
#include <thread>
#include <sys/mman.h>
#include <sys/fcntl.h>
#include <unistd.h>
#include "logger/logger.h"

#define INVALID_32 0xFFFFFFFF
#define INVALID_64 0xFFFFFFFFFFFFFFFF
#define UCM_SHM_FILENAME "/ucmlocalstore.shm"
#define UCM_SHM_MAGIC 0x1122334455667788

namespace UCM {

using Byte = uint8_t;

struct alignas(1) CacheIndex {
    uint32_t idx;
    uint32_t version;

    CacheIndex() : idx{INVALID_32}, version{0} {}
    CacheIndex(uint32_t idx, uint32_t version) : idx{idx}, version{version} {}

    bool InValid() const { return this->idx == INVALID_32; }
};
static_assert(sizeof(CacheIndex) == 8);

struct alignas(1) CacheId {
    std::atomic<uint64_t> le;
    std::atomic<uint64_t> ri;

    CacheId() : le{INVALID_64}, ri{INVALID_64} {}

    void Init()
    {
        this->le.store(INVALID_64, std::memory_order_release);
        this->ri.store(INVALID_64, std::memory_order_release);
    }

    void Set(std::string_view cache_id)
    {
        uint64_t cache_id_le = INVALID_64;
        uint64_t cache_id_ri = INVALID_64;
        std::memcpy(&cache_id_le, cache_id.data(), 8);
        std::memcpy(&cache_id_ri, cache_id.data() + 8, 8);
        this->le.store(cache_id_le, std::memory_order_release);
        this->ri.store(cache_id_ri, std::memory_order_release);
    }

    bool Equal(uint64_t cache_id_le, uint64_t cache_id_ri)
    {
        return (this->le.load(std::memory_order_acquire) == cache_id_le &&
                this->ri.load(std::memory_order_acquire) == cache_id_ri);
    }
};
static_assert(sizeof(CacheId) == 16);

struct alignas(1) CacheRefCnt {
    uint32_t val;
    uint32_t version;

    CacheRefCnt() : val{0}, version{0} {}
    CacheRefCnt(uint32_t val, uint32_t version) : val{val}, version{version} {}
};
static_assert(sizeof(CacheRefCnt) == 8);

enum class CacheStatus : uint64_t {
    ACTIVE = 0,
    INACTIVE,
    READING,
    WRITING,
    EVICTING,
    PINNING,
};
static_assert(sizeof(CacheStatus) == 8);

struct alignas(1) CacheInfo {
    CacheId id;
    std::atomic<CacheIndex> prev;
    std::atomic<CacheIndex> next;
    std::atomic<uint64_t> access_time;
    std::atomic<CacheRefCnt> ref_cnt;
    std::atomic<CacheStatus> status;
    uint32_t pinning_idx;
    Byte padding[4];

    void Init(uint32_t pinning_idx)
    {
        this->id.Init();
        this->prev.store(CacheIndex{}, std::memory_order_release);
        this->next.store(CacheIndex{}, std::memory_order_release);
        this->access_time.store(INVALID_64, std::memory_order_release);
        this->ref_cnt.store(CacheRefCnt{}, std::memory_order_release);
        this->status.store(CacheStatus::INACTIVE, std::memory_order_release);
        this->pinning_idx = pinning_idx;
    }
};
static_assert(sizeof(CacheInfo) == 64);

struct alignas(1) Cache {
    CacheInfo info;
    Byte data[];

    void Init(uint32_t pinning_idx, uint32_t cache_size)
    {
        this->info.Init(pinning_idx);
    }
};
static_assert(sizeof(Cache) == 64);

struct alignas(1) CacheNumber {
    uint32_t val;
    uint32_t version;

    CacheNumber() : val{0}, version{0} {}
    CacheNumber(uint32_t val, uint32_t version) : val{val}, version{version} {}
};
static_assert(sizeof(CacheNumber) == 8);

struct alignas(1) CacheHeader {
    std::atomic<uint64_t> magic;
    std::atomic<CacheNumber> len;
    std::atomic<CacheIndex> head;
    std::atomic<CacheIndex> tail;
    uint32_t cap;
    uint32_t cache_size;
    Byte reserved[4056];
    Cache caches[];

    void Init(uint32_t cap, uint32_t cache_size)
    {
        this->cap = cap;
        this->len.store(CacheNumber{}, std::memory_order_release);
        this->cache_size = cache_size;
        this->head.store(CacheIndex{0, 0}, std::memory_order_release);
        this->tail.store(CacheIndex{}, std::memory_order_release);
        for (uint32_t i = 0; i < cap; ++i) {
            Cache* cache = reinterpret_cast<Cache*>(reinterpret_cast<Byte*>(this->caches) +
                                                    i * (sizeof(CacheInfo) + cache_size));
            cache->Init(i, cache_size);
        }
        this->magic.store(UCM_SHM_MAGIC, std::memory_order_release);
    }

    Cache* CachesAt(uint32_t i)
    {
        return reinterpret_cast<Cache*>(reinterpret_cast<Byte*>(this->caches) +
                                        i * (sizeof(CacheInfo) + cache_size));
    }

    void push_front(Cache* new_head_cache)
    {
        auto old_tail = CacheIndex{};
        auto new_tail = CacheIndex{};
        do {
            old_tail = this->tail.load(std::memory_order_acquire);
            if (!old_tail.InValid()) {
                break;
            }
            new_tail = CacheIndex{new_head_cache->info.pinning_idx, old_tail.version + 1};
        } while(this->tail.compare_exchange_strong(old_tail, new_tail, std::memory_order_acq_rel));

        auto old_prev = new_head_cache->info.prev.load(std::memory_order_acquire);
        auto new_prev = CacheIndex{INVALID_32, old_prev.version + 1};
        new_head_cache->info.prev.store(new_prev, std::memory_order_release);

        auto old_head = CacheIndex{};
        auto new_head = CacheIndex{};
        do {
            old_head = this->head.load(std::memory_order_release);
            new_head = CacheIndex{new_head_cache->info.pinning_idx, old_head.version + 1};
        } while(!this->head.compare_exchange_strong(old_head, new_head, std::memory_order_acq_rel));

        auto old_head_cache = this->CachesAt(old_head.idx);
        old_prev = old_head_cache->info.prev.load(std::memory_order_acquire);
        new_prev = CacheIndex{new_head_cache->info.pinning_idx, old_prev.version + 1};
        old_head_cache->info.prev.store(new_prev, std::memory_order_release);

        auto old_next = new_head_cache->info.next.load(std::memory_order_acquire);
        auto new_next = CacheIndex{old_head_cache->info.pinning_idx, old_next.version + 1};
        new_head_cache->info.next.store(new_next, std::memory_order_release);
    }

    void pop_back(Cache* new_tail_cache)
    {
        auto old_tail = this->tail.load(std::memory_order_acquire);
        auto new_tail = CacheIndex{new_tail_cache->info.pinning_idx, old_tail.version + 1};
        this->tail.store(new_tail, std::memory_order_release);
    }
};
static_assert(sizeof(CacheHeader) == 4096);

LRUCache::LRUCache() : _h{nullptr}, _f{File::Make(UCM_SHM_FILENAME)} {}

LRUCache::~LRUCache()
{
    if (this->_h != nullptr) {
        File::MUnMap(reinterpret_cast<void*>(this->_h),
                     sizeof(CacheHeader) + this->_h->cap * (sizeof(CacheInfo) + this->_h->cache_size));
    }
}

Status LRUCache::MappingCheck(uint64_t shm_size)
{
    auto status = Status::OK;

    status = this->_f->ShmOpen(OpenFlag::RDWR);
    if (status != Status::OK) {
        return status;
    }

    FileStat stat;
    do {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        status = this->_f->Stat(stat);
        if (status != Status::OK) {
            return status;
        }
    } while(static_cast<uint64_t>(stat.st_size) != shm_size);

    status = this->_f->MMap(reinterpret_cast<void**>(&(this->_h)), shm_size, PROT_READ | PROT_WRITE, MAP_SHARED);
    if (status != Status::OK) {
        return status;
    }

    while (this->_h->magic.load(std::memory_order_acquire) != UCM_SHM_MAGIC) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return status;
}

Status LRUCache::MappingSet(uint64_t shm_size, uint32_t cap, uint32_t cache_size)
{
    auto status = this->_f->MMap(reinterpret_cast<void**>(&(this->_h)), shm_size, PROT_READ | PROT_WRITE, MAP_SHARED);
    if (status != Status::OK) {
        return status;
    }
    this->_h->Init(cap, cache_size);
    return status;
}

Status LRUCache::Init(uint32_t cap, uint32_t cache_size)
{
    uint64_t shm_size = sizeof(CacheHeader) + cap * (sizeof(CacheInfo) + cache_size);

    auto status = this->_f->ShmOpen(OpenFlag::RDWR | OpenFlag::CREAT | OpenFlag::EXCL);
    if (status != Status::OK) {
        if (status == Status::EXIST) {
            return this->MappingCheck(shm_size);
        } else {
            return status;
        }
    }

    status = this->_f->Truncate(shm_size);
    if (status != Status::OK) {
        return status;
    }

    return this->MappingSet(shm_size, cap, cache_size);
}

void LRUCache::Evict()
{
    if (this->_h->len.load(std::memory_order_acquire).val != this->_h->cap) {
        return;
    }

    auto old_tail_cache = this->_h->CachesAt(this->_h->tail.load(std::memory_order_acquire).idx);
    auto expected_status = CacheStatus::ACTIVE;
    auto ok = old_tail_cache->info.status.compare_exchange_strong(expected_status,
                                                                  CacheStatus::EVICTING,
                                                                  std::memory_order_acq_rel);
    if (!ok) {
        return;
    }
    auto new_tail_cache = this->_h->CachesAt(old_tail_cache->info.prev.load(std::memory_order_acq_rel).idx);
    ok = new_tail_cache->info.status.compare_exchange_strong(expected_status,
                                                             CacheStatus::PINNING,
                                                             std::memory_order_acq_rel);
    if (!ok) {
        old_tail_cache->info.status.store(CacheStatus::ACTIVE, std::memory_order_release);
        return;
    }

    this->_h->pop_back(new_tail_cache);

    old_tail_cache->info.status.store(CacheStatus::INACTIVE, std::memory_order_release);

    auto old_len = this->_h->len.load(std::memory_order_acquire);
    this->_h->len.store(CacheNumber{old_len.val - 1, old_len.version + 1}, std::memory_order_release);

    new_tail_cache->info.status.store(CacheStatus::ACTIVE, std::memory_order_release);
}

Status LRUCache::Alloc(std::string_view cache_id, void** cache_data)
{
    *cache_data = nullptr;
    if (cache_id.size() != 16) { return Status::INVALID_PARAM; }

    if (this->_h->len.load(std::memory_order_acquire).val == this->_h->cap) {
        std::thread t([this]() { this->Evict(); });
        t.detach();
        return Status::BUSY;
    }

    for (uint32_t i = 0; i < this->_h->cap; ++i) {
        auto cache = this->_h->CachesAt(i);

        auto expected_status = CacheStatus::INACTIVE;
        auto ok = cache->info.status.compare_exchange_strong(expected_status,
                                                             CacheStatus::WRITING,
                                                             std::memory_order_acq_rel);
        if (!ok) {
            continue;
        }
        auto old_len = CacheNumber{};
        auto new_len = CacheNumber{};
        do {
            old_len = this->_h->len.load(std::memory_order_acquire);
            new_len = CacheNumber{old_len.val + 1, old_len.version + 1};
        } while(!this->_h->len.compare_exchange_strong(old_len, new_len, std::memory_order_acq_rel));
        cache->info.id.Set(cache_id);
        *cache_data = cache->data;

        return Status::OK;
    }

    return Status::BUSY;
}

Status LRUCache::Find(std::string_view cache_id, void** cache_data)
{
    *cache_data = nullptr;
    if (cache_id.size() != 16) { return Status::INVALID_PARAM; }
    uint64_t cache_id_le = INVALID_64;
    uint64_t cache_id_ri = INVALID_64;
    std::memcpy(&cache_id_le, cache_id.data(), 8);
    std::memcpy(&cache_id_ri, cache_id.data() + 8, 8);

    if (this->_h->len.load(std::memory_order_acquire).val == 0) {
        return Status::EMPTY;
    }

    for (uint32_t i = 0; i < this->_h->cap; ++i) {
        auto cache = this->_h->CachesAt(i);
        if (cache->info.id.Equal(cache_id_le, cache_id_ri)) {
            while (true) {
                auto cache_status = cache->info.status.load(std::memory_order_acquire);
                if (cache_status == CacheStatus::INACTIVE) {
                    return Status::NOT_EXIST;
                }
                if (cache_status == CacheStatus::EVICTING ||
                    cache_status == CacheStatus::WRITING  ||
                    cache_status == CacheStatus::PINNING) {

                    return Status::BUSY;
                }
                auto ok = true;
                if (cache_status == CacheStatus::ACTIVE) {
                    ok = cache->info.status.compare_exchange_strong(cache_status,
                                                                    CacheStatus::READING,
                                                                    std::memory_order_acq_rel);
                    if (!ok) {
                        continue;
                    }
                }
                auto old_ref_cnt = cache->info.ref_cnt.load(std::memory_order_acquire);
                auto new_ref_cnt = CacheRefCnt{old_ref_cnt.val + 1, old_ref_cnt.version + 1};
                ok = cache->info.ref_cnt.compare_exchange_strong(old_ref_cnt, new_ref_cnt,
                                                                 std::memory_order_acq_rel);
                if (!ok) {
                    continue;
                }
                cache->info.status.store(CacheStatus::READING, std::memory_order_release);
                *cache_data = cache->data;
                return Status::OK;
            }
        }
    }

    return Status::NOT_EXIST;
}

void LRUCache::AllocCommit(void* cache_data)
{
    auto cache = reinterpret_cast<Cache*>(reinterpret_cast<Byte*>(cache_data) - sizeof(CacheInfo));
    this->_h->push_front(cache);
    cache->info.status.store(CacheStatus::ACTIVE, std::memory_order_release);
}

void LRUCache::FindCommit(void* cache_data)
{
    auto cache = reinterpret_cast<Cache*>(reinterpret_cast<Byte*>(cache_data) - sizeof(CacheInfo));

    this->_h->push_front(cache);

    auto old_ref_cnt = CacheRefCnt{};
    auto new_ref_cnt = CacheRefCnt{};
    do {
        old_ref_cnt = cache->info.ref_cnt.load(std::memory_order_acquire);
        new_ref_cnt = CacheRefCnt{old_ref_cnt.val - 1, old_ref_cnt.version + 1};
    } while(!cache->info.ref_cnt.compare_exchange_strong(old_ref_cnt, new_ref_cnt, std::memory_order_acq_rel));

    if (cache->info.ref_cnt.load(std::memory_order_acquire).val == 0) {
        cache->info.status.store(CacheStatus::ACTIVE, std::memory_order_release);
    }
}

} // namespace UCM