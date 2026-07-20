/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_BUFFER_MANAGER_H
#define UNIFIEDCACHE_DRAM_STORE_CC_BUFFER_MANAGER_H

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "buffer.h"
#include "core/transport.h"
#include "logger/logger.h"
#include "pool/buffer_pool.h"
#include "status/status.h"

namespace UC::DramPool {

/**
 * @brief Owns a set of BufferPool instances keyed by size.
 *
 * For each (slotSize, slotNum) pair, a BufferPool is constructed and initialized
 * with MemoryType::HOST. The pool name is derived from the size as
 * "buffer_pool_<size>".
 */
class BufferManager {
public:
    explicit BufferManager(const std::vector<std::pair<std::size_t, std::size_t>>& slots)
    {
        memoryRegions_.reserve(slots.size());
        for (auto [slotSize, slotNum] : slots) {
            if (pools_.find(slotSize) != pools_.end()) { continue; }
            auto pool = std::make_unique<BufferPool>();
            auto st = pool->Init("buffer_pool_" + std::to_string(slotSize),
                                 BufferPool::MemoryType::HOST, slotSize, slotNum);
            if (!st.Success()) {
                UC_ERROR("BufferManager: Init pool for size {} failed, status {}.", slotSize,
                         st.ToString());
                throw std::runtime_error("BufferPool Init failed for size " +
                                         std::to_string(slotSize));
            }
            transport::MemoryRegion region;
            region.addr = pool->GetLocalAddr();
            region.length = pool->GetTotalSize();
            region.type = transport::MemoryType::Host;
            memoryRegions_.push_back(region);
            pools_.emplace(slotSize, std::move(pool));
        }
    }

    ~BufferManager() { Reset(); }

    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;

    /**
     * @brief Release every underlying BufferPool and clear cached memory regions.
     */
    void Reset() noexcept
    {
        for (auto& [_, pool] : pools_) {
            if (pool != nullptr) { pool->Reset(); }
        }
        pools_.clear();
        memoryRegions_.clear();
    }

    /**
     * @brief Look up the BufferPool registered for the given size.
     * @return Pointer to the BufferPool, or nullptr if the size was not
     *         registered at construction.
     */
    BufferPool* GetPool(std::size_t size)
    {
        auto it = pools_.find(size);
        return it == pools_.end() ? nullptr : it->second.get();
    }

    /**
     * @brief Memory regions registered for each pool.
     */
    const std::vector<transport::MemoryRegion>& MemoryRegions() const noexcept
    {
        return memoryRegions_;
    }

    /**
     * @brief Allocate a slot from the pool matching the requested size.
     * @param size [in] Requested buffer size, selects the pool.
     * @param buf [out] On success, buf is filled from the allocated slot.
     * @return Status::OK() on success; Status::NotFound() if no pool is
     *         registered for the requested size; otherwise the status
     *         returned by BufferPool::Allocate.
     */
    Status Allocate(std::size_t size, Buffer& buf)
    {
        auto* pool = GetPool(size);
        if (pool == nullptr) {
            UC_ERROR("BufferManager::Allocate: no pool registered for size {}.", size);
            return Status::NotFound();
        }
        BufferPool::Slot slot;
        auto st = pool->Allocate(slot);
        if (!st.Success()) { return st; }
        buf.length = slot.length;
        buf.slot = slot.slot_index;
        buf.addr = slot.local_addr;
        return Status::OK();
    }

    /**
     * @brief Free a previously allocated slot from the pool matching the
     *        given size.
     * @param size [in] Buffer size, selects the pool.
     * @param slot [in] Slot index returned by Allocate.
     * @return Status::OK() on success; Status::NotFound() if no pool is
     *         registered for the requested size; otherwise the status
     *         returned by BufferPool::Free.
     */
    Status Free(std::size_t size, std::uint32_t slot)
    {
        auto* pool = GetPool(size);
        if (pool == nullptr) {
            UC_ERROR("BufferManager::Free: no pool registered for size {}.", size);
            return Status::NotFound();
        }
        return pool->Free(slot);
    }

private:
    std::unordered_map<std::size_t, std::unique_ptr<BufferPool>> pools_;
    std::vector<transport::MemoryRegion> memoryRegions_;
};

}  // namespace UC::DramPool

#endif
