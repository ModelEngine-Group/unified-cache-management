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
#include "index_queue.h"
#include <atomic>
#include <thread>
#include <sys/mman.h>

#define UCM_INDEX_QUEUE_SHM_MAGIC 0x89ABCDEF

namespace UCM {

struct Index {
    uint32_t val;
    uint32_t version;

    Index() : val{0}, version{0} {}
    Index(const uint32_t val, const uint32_t version) : val{val}, version{version} {}
};

struct Length {
    uint32_t val;
    uint32_t version;

    Length() : val{0}, version{0} {}
    Length(const uint32_t val, const uint32_t version) : val{val}, version{version} {}

    bool operator==(const uint32_t val) const {
        return this->val == val;
    }
};

struct IndexQueueHeader {
    std::atomic<uint32_t> magic;
    std::atomic<uint32_t> cap;
    std::atomic<Length>   len;
    std::atomic<Index>    head;
    std::atomic<Index>    tail;
    std::atomic<uint32_t> vals[];

    void Initialize(const uint32_t cap)
    {
        this->cap.store(cap, std::memory_order_release);
        this->len.store(Length{cap, 0}, std::memory_order_release);
        this->head.store(Index{0, 0}, std::memory_order_release);
        this->tail.store(Index{cap - 1, 0}, std::memory_order_release);
        for (uint32_t i = 0; i < cap; ++i) {
           vals[i].store(i, std::memory_order_release);
        }
        this->magic.store(UCM_INDEX_QUEUE_SHM_MAGIC, std::memory_order_release);
    }

    bool Push(const uint32_t val)
    {
        auto old_len = Length{};
        auto new_len = Length{};
        do {
            old_len = this->len.load(std::memory_order_acquire);
            if (old_len == this->cap.load(std::memory_order_relaxed)) { return false; }
            new_len = Length{old_len.val + 1, old_len.version + 1};
        } while (!this->len.compare_exchange_strong(old_len, new_len, std::memory_order_acq_rel));

        auto old_tail = Index{};
        auto new_tail = Index{};
        do {
            old_tail = this->tail.load(std::memory_order_acquire);
            new_tail = Index{
                (old_tail.val + 1) % this->cap.load(std::memory_order_relaxed),
                old_tail.version + 1
            };
        } while (!this->tail.compare_exchange_strong(old_tail, new_tail, std::memory_order_acq_rel));

        this->vals[old_tail.val].store(val, std::memory_order_release);

        return true;
    }

    std::optional<uint32_t> Pop()
    {
        auto old_len = Length{};
        auto new_len = Length{};
        do {
            old_len = this->len.load(std::memory_order_acquire);
            if (old_len == 0) { return {}; }
            new_len = Length{old_len.val - 1, old_len.version + 1};
        } while (!this->len.compare_exchange_strong(old_len, new_len, std::memory_order_acq_rel));

        auto old_head = Index{};
        auto new_head = Index{};
        do {
            old_head = this->head.load(std::memory_order_acquire);
            new_head = Index{
                (old_head.val + 1) % this->cap.load(std::memory_order_relaxed),
                old_head.version + 1
            };
        } while (!this->head.compare_exchange_strong(old_head, new_head, std::memory_order_acq_rel));

        return {this->vals[old_head.val].load(std::memory_order_acquire)};
    }
};


IndexQueue::IndexQueue(std::string_view filename) : _h{nullptr}, _f{File::Make(filename)} {}

IndexQueue::~IndexQueue()
{
    if (this->_h != nullptr) {
        File::MUnMap(
            this->_h,
            static_cast<uint64_t>(sizeof(IndexQueueHeader)) +
            static_cast<uint64_t>(this->_h->cap.load(std::memory_order_relaxed)) *
            static_cast<uint64_t>(sizeof(std::atomic<uint32_t>))
        );
    }
}

Status IndexQueue::Initialize(const uint32_t queue_size)
{
    uint64_t shm_size = static_cast<uint64_t>(sizeof(IndexQueueHeader)) +
                        static_cast<uint64_t>(queue_size) * static_cast<uint64_t>(sizeof(std::atomic<uint32_t>));

    auto s = this->_f->ShmOpen(OpenFlag::RDWR | OpenFlag::CREAT | OpenFlag::EXCL);
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

    return this->MappingInitialize(shm_size, queue_size);
}

bool IndexQueue::Push(const uint32_t val)
{
    return this->_h->Push(val);
}

std::optional<uint32_t> IndexQueue::Pop()
{
    return this->_h->Pop();
}

Status IndexQueue::MappingCheck(const uint64_t shm_size)
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

    while (this->_h->magic.load(std::memory_order_acquire) != UCM_INDEX_QUEUE_SHM_MAGIC) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return s;
}

Status IndexQueue::MappingInitialize(const uint64_t shm_size, const uint32_t queue_size)
{
    auto s = this->_f->MMap(reinterpret_cast<void**>(&(this->_h)), shm_size, PROT_READ | PROT_WRITE, MAP_SHARED);
    if (s != Status::OK) {
        return s;
    }
    this->_h->Initialize(queue_size);
    return s;
}

} // namespace UCM