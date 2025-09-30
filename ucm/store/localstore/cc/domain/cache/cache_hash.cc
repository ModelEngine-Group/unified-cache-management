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
#include "cache_hash.h"
#include <atomic>
#include <pthread.h>

namespace UC {

static constexpr uint32_t BucketNumber = 17683;
static constexpr uint32_t Magic = (('C' << 16) | ('h' << 8) | 1);

struct Key {
    uint64_t k1, k2;
    size_t off;
    Key() { this->Init(); }
    Key(const std::string& id, const size_t off) { this->Fill(id, off); }
    bool operator==(const Key& k) const { return k1 == k.k1 && k2 == k.k2 && off == k.off; }
    void Init() { this->k1 = this->k2 = this->off = 0; }
    void Fill(const std::string& id, const size_t off)
    {
        auto idPair = static_cast<const uint64_t*>(static_cast<const void*>(id.data()));
        this->k1 = idPair[0];
        this->k2 = idPair[1];
        this->off = off;
    }
    uint32_t Hash()
    {
        static std::hash<size_t> hasher{};
        return (hasher(k1) | hasher(k2) | hasher(off)) % BucketNumber;
    }
};

struct Node {
    Key key;
    uint32_t prev;
    uint32_t next;
    std::atomic<uint64_t> ref;
    uint64_t tp;
    void Init()
    {
        this->key.Init();
        this->prev = CacheHash::npos;
        this->next = CacheHash::npos;
        this->ref = 0;
    }
};

struct Bucket {
    pthread_mutex_t mutex;
    uint32_t head;
    uint32_t tail;
    void Init()
    {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&this->mutex, &attr);
        pthread_mutexattr_destroy(&attr);
        this->head = CacheHash::npos;
        this->tail = CacheHash::npos;
    }
    void Lock() { pthread_mutex_lock(&this->mutex); }
    void Unlock() { pthread_mutex_unlock(&this->mutex); }
};

struct Header {
    uint32_t magic;
    uint32_t capacity;
    std::atomic<uint64_t> tp;
    uint64_t reserved[7];
    Bucket buckets[BucketNumber];
    Node nodes[0];
};

inline auto HeaderPtr(void* addr) { return (Header*)addr; }

inline void InsertNode2Bucket(Header* header, Bucket* bucket, const uint32_t index)
{
    auto slot = header->nodes + index;
    slot->prev = CacheHash::npos;
    slot->next = bucket->head;
    if (bucket->head != CacheHash::npos) {
        auto next = header->nodes + bucket->head;
        next->prev = index;
    }
    bucket->head = index;
    if (bucket->tail == CacheHash::npos) { bucket->tail = index; }
}

inline void MoveNode2BucketHead(Header* header, Bucket* bucket, const uint32_t index)
{
    if (bucket->head == index) { return; }
    auto slot = header->nodes + index;
    if (bucket->tail == index) {
        auto tail = header->nodes + slot->prev;
        tail->next = CacheHash::npos;
        bucket->tail = slot->prev;
        slot->prev = CacheHash::npos;
    } else {
        auto prev = header->nodes + slot->prev;
        auto next = header->nodes + slot->next;
        prev->next = slot->next;
        next->prev = slot->prev;
    }
    auto head = header->nodes + bucket->head;
    head->prev = index;
    slot->next = bucket->head;
    bucket->head = index;
}

inline void RemoveNodeFromBucket(Header* header, Bucket* bucket, const uint32_t index)
{
    auto slot = header->nodes + index;
    if (bucket->head == index && bucket->tail == index) {
        bucket->head = bucket->tail = CacheHash::npos;
        return;
    }
    if (bucket->head != index && bucket->tail != index) {
        auto prev = header->nodes + slot->prev;
        auto next = header->nodes + slot->next;
        prev->next = slot->next;
        next->prev = slot->prev;
        return;
    }
    if (bucket->head == index) {
        bucket->head = slot->next;
        auto head = header->nodes + slot->next;
        head->prev = CacheHash::npos;
        return;
    }
    if (bucket->tail == index) {
        bucket->tail = slot->prev;
        auto tail = header->nodes + slot->prev;
        tail->next = CacheHash::npos;
        return;
    }
}

size_t CacheHash::MemorySize() const noexcept
{
    return sizeof(Header) * sizeof(Node) * this->capacity_;
}

void CacheHash::Setup(void* addr) noexcept
{
    this->addr_ = addr;
    auto header = HeaderPtr(this->addr_);
    if (header->magic == Magic) { return; }
    header->capacity = this->capacity_;
    header->tp.store(0, std::memory_order_relaxed);
    auto reservedSize = sizeof(header->reserved) / sizeof(*header->reserved);
    std::fill_n(header->reserved, reservedSize, 0);
    for (uint32_t i = 0; i < BucketNumber; i++) { header->buckets[i].Init(); }
    for (uint32_t i = 0; i < header->capacity; i++) { header->nodes[i].Init(); }
    header->magic = Magic;
    return;
}

void CacheHash::Insert(const std::string& id, const size_t offset, const uint32_t index) noexcept
{
    auto header = HeaderPtr(this->addr_);
    auto slot = header->nodes + index;
    slot->key.Fill(id, offset);
    auto bucket = header->buckets + slot->key.Hash();
    bucket->Lock();
    slot->ref.store(1, std::memory_order_relaxed);
    slot->tp = header->tp.fetch_add(1);
    InsertNode2Bucket(header, bucket, index);
    bucket->Unlock();
}

uint32_t CacheHash::Find(const std::string& id, const size_t offset) noexcept
{
    Key key{id, offset};
    auto header = HeaderPtr(this->addr_);
    auto bucket = header->buckets + key.Hash();
    bucket->Lock();
    auto pos = bucket->head;
    while (pos != npos) {
        auto slot = header->nodes + pos;
        if (slot->key == key) {
            slot->ref++;
            slot->tp = header->tp.fetch_add(1);
            MoveNode2BucketHead(header, bucket, pos);
            break;
        }
        pos = slot->next;
    }
    bucket->Unlock();
    return pos;
}

void CacheHash::PutRef(const uint32_t index) noexcept
{
    auto header = HeaderPtr(this->addr_);
    if (index >= header->capacity) { return; }
    auto slot = header->nodes + index;
    auto ref = slot->ref.load(std::memory_order_acquire);
    while (ref > 0) {
        auto desired = ref - 1;
        if (slot->ref.compare_exchange_weak(ref, desired, std::memory_order_acq_rel)) { break; }
        ref = slot->ref.load(std::memory_order_acquire);
    }
}

void CacheHash::PutRef(const std::string& id, const size_t offset) noexcept
{
    Key key{id, offset};
    auto header = HeaderPtr(this->addr_);
    auto bucket = header->buckets + key.Hash();
    bucket->Lock();
    auto pos = bucket->head;
    while (pos != npos) {
        auto slot = header->nodes + pos;
        if (slot->key == key) {
            this->PutRef(pos);
            break;
        }
        pos = slot->next;
    }
    bucket->Unlock();
}

void CacheHash::Remove(const std::string& id, const size_t offset) noexcept
{
    Key key{id, offset};
    auto header = HeaderPtr(this->addr_);
    auto bucket = header->buckets + key.Hash();
    bucket->Lock();
    auto pos = bucket->head;
    while (pos != npos) {
        auto slot = header->nodes + pos;
        if (slot->key == key) {
            RemoveNodeFromBucket(header, bucket, pos);
            break;
        }
        pos = slot->next;
    }
    bucket->Unlock();
}

uint32_t CacheHash::Evict() noexcept
{
    auto header = HeaderPtr(this->addr_);
    auto iBucket = npos;
    auto pos = npos;
    auto tp = (uint64_t)(-1);
    for (uint32_t i = 0; i < BucketNumber; i++) {
        auto bucket = header->buckets + i;
        bucket->Lock();
        if (bucket->tail != npos) {
            auto slot = header->nodes + bucket->tail;
            if (slot->ref == 0 && tp > slot->tp) {
                iBucket = i;
                pos = bucket->tail;
                tp = slot->tp;
            }
        }
        bucket->Unlock();
    }
    if (iBucket == npos) { return npos; }
    auto bucket = header->buckets + iBucket;
    bucket->Lock();
    auto slot = header->nodes + pos;
    if (bucket->tail != pos || slot->ref != 0) {
        pos = npos;
    } else {
        RemoveNodeFromBucket(header, bucket, pos);
    }
    bucket->Unlock();
    return pos;
}

} // namespace UC
