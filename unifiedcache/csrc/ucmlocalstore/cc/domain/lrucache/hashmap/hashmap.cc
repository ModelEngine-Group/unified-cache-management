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
#include <sys/mman.h>
#include <pthread.h>

#define UCM_HASHMAP_SHM_PRIME 11273
#define UCM_HASHMAP_INVALID_64 0xFFFFFFFFFFFFFFFF
#define UCM_HASHMAP_SHM_MAGIC 0x8877665544332211
#define UCM_HASHMAP_INDEX_QUEUE_SHM_FILENAME "/ucm_hashmap_index_queue"

namespace UCM {

struct Key {
    uint64_t l;
    uint64_t r;

    Key() : l{UCM_HASHMAP_INVALID_64}, r{UCM_HASHMAP_INVALID_64} {}
    Key(std::string_view key)
    {
        const uint64_t* p = reinterpret_cast<const uint64_t*>(key.data());
        this->l = p[0];
        this->r = p[1];
    }

    bool operator==(const Key& other)
    {
        return (this->l == other.l && this->r == other.r);
    }
};
static_assert(sizeof(Key) == 16);

struct Node {
    Key key;
    uint64_t val;
    uint64_t next;

    void Initialize()
    {
        this->key = Key{};
        this->val = UCM_HASHMAP_INVALID_64;
        this->next = UCM_HASHMAP_INVALID_64;
    }
};

struct Bucket {
    pthread_mutex_t mtx;
    uint64_t head;

    void Initialize()
    {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&mtx, &attr);
        pthread_mutexattr_destroy(&attr);

        this->head = UCM_HASHMAP_INVALID_64;
    }
};

struct Length {
    uint32_t val;
    uint32_t version;

    Length() : val{0}, version{0} {}
    Length(uint64_t val, uint32_t version) : val{static_cast<uint32_t>(val)}, version{version} {}

    bool operator==(const uint64_t val) {
        return (static_cast<uint64_t>(this->val) == val);
    }
};


struct HashMapHeader {
    std::atomic<uint64_t> magic;
    std::atomic<uint64_t> cap;
    std::atomic<Length> len;
    Bucket buckets[UCM_HASHMAP_SHM_PRIME];
    Node nodes[];

    void Initialize(const uint64_t cap)
    {
        this->cap.store(cap, std::memory_order_release);
        this->len.store(Length{}, std::memory_order_release);
        for (uint64_t i = 0; i < UCM_HASHMAP_SHM_PRIME; ++i) {
            this->buckets[i].Initialize();
        }
        for (uint64_t i = 0; i < cap; ++i) {
            this->nodes[i].Initialize();
        }
        this->magic.store(UCM_HASHMAP_SHM_MAGIC, std::memory_order_release);
    }

    uint64_t Hash(std::string_view key)
    {
        const uint64_t* p = reinterpret_cast<const uint64_t*>(key.data());
        uint64_t hash = p[0] ^ p[1];
        hash ^= (hash >> 20) ^ (hash >> 12);
        return hash % UCM_HASHMAP_SHM_PRIME;
    }

    void Insert(std::string_view key, uint64_t val, uint64_t pinning_idx)
    {
        auto old_len = Length{};
        auto new_len = Length{};
        do {
            old_len = this->len.load(std::memory_order_acquire);
            if (old_len == this->cap.load(std::memory_order_relaxed)) { return; }
            new_len = Length{old_len.val + 1, old_len.version + 1};
        } while (!this->len.compare_exchange_strong(old_len, new_len, std::memory_order_acq_rel));

        uint64_t i = this->Hash(key);

        pthread_mutex_lock(&(this->buckets[i].mtx));

        this->nodes[pinning_idx].key = Key{key};
        this->nodes[pinning_idx].val = val;
        this->nodes[pinning_idx].next = UCM_HASHMAP_INVALID_64;

        if (this->buckets[i].head == UCM_HASHMAP_INVALID_64) {
            this->buckets[i].head = pinning_idx;
            pthread_mutex_unlock(&(this->buckets[i].mtx));
            return;
        }

        this->nodes[pinning_idx].next = this->buckets[i].head;
        this->buckets[i].head = pinning_idx;
        pthread_mutex_unlock(&(this->buckets[i].mtx));
    }

    bool Find(std::string_view key, uint64_t& val)
    {
        uint64_t i = this->Hash(key);
        auto target = Key{key};

        pthread_mutex_lock(&(this->buckets[i].mtx));

        uint64_t pinning_idx = this->buckets[i].head;
        while (pinning_idx != UCM_HASHMAP_INVALID_64) {
            if (this->nodes[pinning_idx].key == target) {
                val = this->nodes[pinning_idx].val;
                pthread_mutex_unlock(&(this->buckets[i].mtx));
                return true;
            }
            pinning_idx = this->nodes[pinning_idx].next;
        }

        pthread_mutex_unlock(&(this->buckets[i].mtx));

        return false;
    }

    uint64_t Remove(std::string_view key)
    {
        uint64_t i = this->Hash(key);
        auto target = Key{key};

        pthread_mutex_lock(&(this->buckets[i].mtx));

        uint64_t pinning_idx = this->buckets[i].head;
        uint64_t prev_idx = UCM_HASHMAP_INVALID_64;
        while (pinning_idx != UCM_HASHMAP_INVALID_64) {
            if (this->nodes[pinning_idx].key == target) {
                if (prev_idx == UCM_HASHMAP_INVALID_64) {
                    this->buckets[i].head = this->nodes[pinning_idx].next;
                } else {
                    this->nodes[prev_idx].next = this->nodes[pinning_idx].next;
                }
                this->nodes[pinning_idx].Initialize();
                pthread_mutex_unlock(&(this->buckets[i].mtx));
                return pinning_idx;
            }
            prev_idx = pinning_idx;
            pinning_idx = this->nodes[pinning_idx].next;
        }
        pthread_mutex_unlock(&(this->buckets[i].mtx));

        return pinning_idx;
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
            this->_h->cap.load(std::memory_order_relaxed) * static_cast<uint64_t>(sizeof(Node))
        );
    }
}

Status HashMap::Initialize(const uint64_t num)
{
    auto status = this->_q.Initialize(num);
    if (status != Status::OK) {
        return status;
    }

    uint64_t shm_cap = static_cast<uint64_t>(sizeof(HashMapHeader)) + num * static_cast<uint64_t>(sizeof(Node));

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

    return this->MappingInitialize(shm_cap, num);
}

Status HashMap::Insert(std::string_view key, uint64_t val)
{
    if (key.size() != 16) { return Status::INVALID_PARAM; }

    if (this->_h->len.load(std::memory_order_acquire) == this->_h->cap.load(std::memory_order_relaxed)) {
        return Status::BUSY;
    }

    uint64_t pinning_idx;
    auto status = this->_q.Pop(pinning_idx);
    if (status != Status::OK) {
        return status;
    }

    this->_h->Insert(key, val, pinning_idx);

    return Status::OK;
}

Status HashMap::Find(std::string_view key, uint64_t& val)
{
    val = UCM_HASHMAP_INVALID_64;
    if (key.size() != 16) { return Status::INVALID_PARAM; }

    if (this->_h->len.load(std::memory_order_acquire) == 0) {
        return Status::EMPTY;
    }

    if (this->_h->Find(key, val)) {
        return Status::OK;
    } else {
        return Status::NOT_EXIST;
    }
}

Status HashMap::Remove(std::string_view key)
{
    if (key.size() != 16) { return Status::INVALID_PARAM; }

    if (this->_h->len.load(std::memory_order_acquire) == 0) {
        return Status::EMPTY;
    }

    auto pinning_idx = this->_h->Remove(key);

    auto status = this->_q.Push(pinning_idx);
    if (status != Status::OK) {
        return status;
    }

    return Status::OK;
}

Status HashMap::MappingCheck(const uint64_t shm_cap)
{
    auto status = Status::OK;

    status = this->_f->ShmOpen(OpenFlag::RDWR);
    if (status != Status::OK) {
        return status;
    }

    FileStat stat;
    do {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        status = this->_f->Stat(stat);
        if (status != Status::OK) {
            return status;
        }
    } while(static_cast<uint64_t>(stat.st_size) != shm_cap);

    status = this->_f->MMap(reinterpret_cast<void**>(&(this->_h)), shm_cap, PROT_READ | PROT_WRITE, MAP_SHARED);
    if (status != Status::OK) {
        return status;
    }

    while (this->_h->magic.load(std::memory_order_acquire) != UCM_HASHMAP_SHM_MAGIC) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return status;
}

Status HashMap::MappingInitialize(const uint64_t shm_cap, const uint64_t num)
{
    auto status = this->_f->MMap(reinterpret_cast<void**>(&(this->_h)), shm_cap, PROT_READ | PROT_WRITE, MAP_SHARED);
    if (status != Status::OK) {
        return status;
    }
    this->_h->Initialize(num);
    return status;
}

} // namespace UCM