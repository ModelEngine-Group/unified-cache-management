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
#include "hashmap.h"
#include <atomic>
#include <thread>
#include <algorithm>
#include <functional>
#include <sys/mman.h>
#include <pthread.h>

#define UCM_HASHMAP_PRIME 17683
#define UCM_HASHMAP_INVALID_32 0xFFFFFFFF
#define UCM_HASHMAP_SHM_MAGIC 0x89ABCDEF
#define UCM_HASHMAP_INDEX_QUEUE_SHM_FILENAME "/ucm_lru_cache_hashmap_index_queue"

namespace UCM {

struct Key {
    char data[16];

    Key() { std::fill(std::begin(this->data), std::end(this->data), static_cast<char>(0)); }

    void Reset() { std::fill(std::begin(this->data), std::end(this->data), static_cast<char>(0)); }

    bool operator==(std::string_view key) const
    {
        return std::equal(std::begin(this->data), std::end(this->data), key.begin());
    }

    Key& operator=(std::string_view key)
    {
        std::copy(key.begin(), key.end(), std::begin(this->data));
        return *this;
    }
};

struct Node {
    Key key;
    uint32_t val;
    uint32_t next;

    void Initialize()
    {
        this->val = UCM_HASHMAP_INVALID_32;
        this->next = UCM_HASHMAP_INVALID_32;
    }

    void Reset()
    {
        this->key.Reset();
        this->val = UCM_HASHMAP_INVALID_32;
        this->next = UCM_HASHMAP_INVALID_32;
    }
};

struct Bucket {
    pthread_mutex_t mtx;
    uint32_t head;
    std::byte padding[4];

    void Initialize()
    {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&mtx, &attr);
        pthread_mutexattr_destroy(&attr);

        this->head = UCM_HASHMAP_INVALID_32;
    }

    void Lock() { pthread_mutex_lock(&this->mtx); }
    void Unlock() { pthread_mutex_unlock(&this->mtx); }
};

struct HashMapHeader {
    std::atomic<uint32_t> magic;
    std::atomic<uint32_t> cap;
    Bucket buckets[UCM_HASHMAP_PRIME];
    Node nodes[];

    void Initialize(const uint32_t cap)
    {
        this->cap.store(cap, std::memory_order_release);
        for (uint32_t i = 0; i < UCM_HASHMAP_PRIME; ++i) {
            this->buckets[i].Initialize();
        }
        for (uint32_t i = 0; i < cap; ++i) {
            this->nodes[i].Initialize();
        }
        this->magic.store(UCM_HASHMAP_SHM_MAGIC, std::memory_order_release);
    }

    uint32_t Hash(std::string_view key)
    {
        std::hash<std::string_view> hasher;
        return static_cast<uint32_t>(hasher(key) % UCM_HASHMAP_PRIME);
    }

    Status Insert(std::string_view key, uint32_t val, const uint32_t position)
    {
        auto s = Status::OK;
        uint32_t i = this->Hash(key);

        this->buckets[i].Lock();

        if (this->buckets[i].head == UCM_HASHMAP_INVALID_32) {
            this->nodes[position].key = key;
            this->nodes[position].val = val;
            this->buckets[i].head = position;
            this->buckets[i].Unlock();
            return s;
        }

        auto current = this->buckets[i].head;
        while (true) {
            if (this->nodes[current].key == key) {
                s = Status::EXIST;
                break;
            }
            if (this->nodes[current].next == UCM_HASHMAP_INVALID_32) {
                this->nodes[position].key = key;
                this->nodes[position].val = val;
                this->nodes[current].next = position;
                break;
            }
            current = this->nodes[current].next;
        }

        this->buckets[i].Unlock();

        return s;
    }

    std::optional<uint32_t> Find(std::string_view key)
    {
        uint32_t i = this->Hash(key);

        this->buckets[i].Lock();

        auto position = this->buckets[i].head;
        while (position != UCM_HASHMAP_INVALID_32) {
            if (this->nodes[position].key == key) {
                auto val = this->nodes[position].val;
                this->buckets[i].Unlock();
                return {val};
            }
            position = this->nodes[position].next;
        }

        this->buckets[i].Unlock();

        return {};
    }

    std::optional<uint32_t> Remove(std::string_view key)
    {
        uint32_t i = this->Hash(key);

        this->buckets[i].Lock();

        auto position = this->buckets[i].head;
        uint32_t prev_position = UCM_HASHMAP_INVALID_32;
        while (position != UCM_HASHMAP_INVALID_32) {
            if (this->nodes[position].key == key) {
                if (prev_position == UCM_HASHMAP_INVALID_32) {
                    this->buckets[i].head = this->nodes[position].next;
                } else {
                    this->nodes[prev_position].next = this->nodes[position].next;
                }

                this->nodes[position].Reset();

                this->buckets[i].Unlock();

                return {position};
            }
            prev_position = position;
            position = this->nodes[position].next;
        }

        this->buckets[i].Unlock();

        return {};
    }
};

HashMap::HashMap(std::string_view filename)
    : _h{nullptr}, _f{File::Make(filename)}, _q{UCM_HASHMAP_INDEX_QUEUE_SHM_FILENAME}
{}

HashMap::~HashMap()
{
    if (this->_h != nullptr) {
        File::MUnMap(
            reinterpret_cast<void*>(this->_h),
            static_cast<uint64_t>(sizeof(HashMapHeader)) +
            static_cast<uint64_t>(this->_h->cap.load(std::memory_order_relaxed)) *
            static_cast<uint64_t>(sizeof(Node))
        );
    }
}

Status HashMap::Initialize(const uint32_t map_size)
{
    auto s = this->_q.Initialize(map_size);
    if (s != Status::OK) {
        return s;
    }

    uint64_t shm_size = static_cast<uint64_t>(sizeof(HashMapHeader)) +
                        static_cast<uint64_t>(map_size) * static_cast<uint64_t>(sizeof(Node));

    s = this->_f->ShmOpen(OpenFlag::RDWR | OpenFlag::CREAT | OpenFlag::EXCL);
    if (s != Status::OK) {
        if (s == Status::EXIST) {
            return this->MappingCheck(shm_size);
        } else {
            return s;
        }
    }

    s = this->_f->Truncate(shm_size);
    if (s != Status::OK) {
        return s;
    }

    return this->MappingInitialize(shm_size, map_size);
}

Status HashMap::Insert(std::string_view key, const uint32_t val)
{
    auto position = this->_q.Pop();
    if (!position.has_value()) {
        return Status::BUSY;
    }

    auto s = this->_h->Insert(key, val, *position);
    if (s == Status::EXIST) {
        this->_q.Push(*position);
    }

    return s;
}

std::optional<uint32_t> HashMap::Find(std::string_view key)
{
    return this->_h->Find(key);
}

void HashMap::Remove(std::string_view key)
{
    auto position = this->_h->Remove(key);
    if (!position.has_value()) {
        return;
    }

    this->_q.Push(*position);
}

Status HashMap::MappingCheck(const uint64_t shm_size)
{
    auto s = Status::OK;

    s = this->_f->ShmOpen(OpenFlag::RDWR);
    if (s != Status::OK) {
        return s;
    }

    FileStat stat;
    do {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        s = this->_f->Stat(stat);
        if (s != Status::OK) {
            return s;
        }
    } while(static_cast<uint64_t>(stat.st_size) != shm_size);

    s = this->_f->MMap(reinterpret_cast<void**>(&(this->_h)), shm_size, PROT_READ | PROT_WRITE, MAP_SHARED);
    if (s != Status::OK) {
        return s;
    }

    while (this->_h->magic.load(std::memory_order_acquire) != UCM_HASHMAP_SHM_MAGIC) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return s;
}

Status HashMap::MappingInitialize(const uint64_t shm_size, const uint32_t map_size)
{
    auto s = this->_f->MMap(reinterpret_cast<void**>(&(this->_h)), shm_size, PROT_READ | PROT_WRITE, MAP_SHARED);
    if (s != Status::OK) {
        return s;
    }
    this->_h->Initialize(map_size);
    return s;
}

} // namespace UCM