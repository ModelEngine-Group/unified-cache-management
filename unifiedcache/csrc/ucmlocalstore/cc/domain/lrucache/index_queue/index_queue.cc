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

#define UCM_INDEX_QUEUE_INVALID_64 0xFFFFFFFFFFFFFFFF
#define UCM_INDEX_QUEUE_SHM_MAGIC 0x1122334455667788

namespace UCM {

struct Index {
    uint32_t idx;
    uint32_t version;

    Index() : idx{0}, version{0} {}
    Index(uint32_t idx, uint32_t version) : idx{idx}, version{version} {}
};
static_assert(sizeof(Index) == 8);

struct Length {
    uint32_t val;
    uint32_t version;

    Length() : val{0}, version{0} {}
    Length(uint32_t val, uint32_t version) : val{val}, version{version} {}
    Length(uint64_t val, uint32_t version) : val{static_cast<uint32_t>(val)}, version{version} {}

    bool operator==(const uint64_t val) {
        return static_cast<uint64_t>(this->val) == val;
    }
};
static_assert(sizeof(Index) == 8);

struct IndexQueueHeader {
    std::atomic<uint64_t> magic;
    std::atomic<uint64_t> cap;
    std::atomic<Length> len;
    std::atomic<Index> head;
    std::atomic<Index> tail;
    std::atomic<uint64_t> pinning_indexes[];

    void Initialize(uint64_t cap)
    {
        this->cap.store(cap, std::memory_order_release);
        this->len.store(Length{}, std::memory_order_release);
        this->head.store(Index{}, std::memory_order_release);
        this->tail.store(Index{}, std::memory_order_release);
        for (uint64_t i = 0; i < cap; ++i) {
           pinning_indexes[i].store(i, std::memory_order_release);
        }
        this->magic.store(UCM_INDEX_QUEUE_SHM_MAGIC, std::memory_order_release);
    }

    Status Push(uint64_t pinning_idx)
    {
        auto old_len = Length{};
        auto new_len = Length{};
        do {
            old_len = this->len.load(std::memory_order_acquire);
            if (old_len == this->cap.load(std::memory_order_relaxed)) { return Status::BUSY; }
            new_len = Length{old_len.val + 1, old_len.version + 1};
        } while (!this->len.compare_exchange_strong(old_len, new_len, std::memory_order_acq_rel));

        auto old_tail = Index{};
        auto new_tail = Index{};
        do {
            old_tail = this->tail.load(std::memory_order_acquire);
            new_tail = Index{
                (old_tail.idx + 1) % static_cast<uint32_t>(this->cap.load(std::memory_order_relaxed)),
                old_tail.version + 1
            };
        } while (!this->tail.compare_exchange_strong(old_tail, new_tail, std::memory_order_acq_rel));

        this->pinning_indexes[old_tail.idx].store(pinning_idx, std::memory_order_release);

        return Status::OK;
    }

    Status Pop(uint64_t& pinning_idx)
    {
        auto old_len = Length{};
        auto new_len = Length{};
        do {
            old_len = this->len.load(std::memory_order_acquire);
            if (old_len == 0) { return Status::EMPTY; }
            new_len = Length{old_len.val - 1, old_len.version + 1};
        } while (!this->len.compare_exchange_strong(old_len, new_len, std::memory_order_acq_rel));

        auto old_head = Index{};
        auto new_head = Index{};
        do {
            old_head = this->head.load(std::memory_order_acquire);
            new_head = Index{
                (old_head.idx + 1) % static_cast<uint32_t>(this->cap.load(std::memory_order_relaxed)),
                old_head.version + 1
            };
        } while (!this->head.compare_exchange_strong(old_head, new_head, std::memory_order_acq_rel));

        pinning_idx = this->pinning_indexes[old_head.idx].load(std::memory_order_acquire);

        return Status::OK;
    }
};
static_assert(sizeof(IndexQueueHeader) == 40);


IndexQueue::IndexQueue(std::string_view filename) : _h{nullptr}, _f{File::Make(filename)} {}

IndexQueue::~IndexQueue()
{
    if (this->_h != nullptr) {
        File::MUnMap(
            this->_h,
            sizeof(IndexQueueHeader) +
            this->_h->cap.load(std::memory_order_relaxed) * sizeof(std::atomic<uint64_t>)
        );
    }
}

Status IndexQueue::Initialize(const uint64_t num)
{
    uint64_t shm_cap = sizeof(IndexQueueHeader) + num * sizeof(std::atomic<uint64_t>);

    auto status = this->_f->ShmOpen(OpenFlag::RDWR | OpenFlag::CREAT | OpenFlag::EXCL);
    if (status != Status::OK) {
        if (status == Status::EXIST) {
            return this->MappingCheck(shm_cap, num);
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

Status IndexQueue::Push(const uint64_t pinning_idx)
{
    return this->_h->Push(pinning_idx);
}

Status IndexQueue::Pop(uint64_t& pinning_idx)
{
    return this->_h->Pop(pinning_idx);
}

Status IndexQueue::MappingCheck(const uint64_t shm_cap)
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

    while (this->_h->magic.load(std::memory_order_acquire) != UCM_INDEX_QUEUE_SHM_MAGIC) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return status;
}

Status IndexQueue::MappingInitialize(const uint64_t shm_cap, const uint64_t num)
{
    auto status = this->_f->MMap(reinterpret_cast<void**>(&(this->_h)), shm_cap, PROT_READ | PROT_WRITE, MAP_SHARED);
    if (status != Status::OK) {
        return status;
    }
    this->_h->Initialize(num);
    return status;
}

} // namespace UCM