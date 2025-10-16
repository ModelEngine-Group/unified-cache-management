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
#ifndef UNIFIEDCACHE_MEMORY_POOL_H
#define UNIFIEDCACHE_MEMORY_POOL_H

#include <cstddef>
#include <cstdlib>
#include <string>
#include <list>
#include <unordered_map>
#include <set>
#include "status/status.h"

namespace UC {

class MemoryPool {
public:
    MemoryPool(uint32_t capacity, uint32_t blockSize)
        : pool_(new char[capacity]),
          capacity_(capacity),
          blockSize_(blockSize),
          slotNum_(capacity / blockSize) {
        if (!pool_) throw std::bad_alloc();
        // 1. 预占满：dummy → 地址 同时写进 addressMap_ 和 LRU
        for (uint32_t i = 0; i < slotNum_; ++i) {
            std::string dummy = "__slot_" + std::to_string(i);
            char* addr = pool_ + i * blockSize_;
            // 填 LRU
            lruList_.push_front(dummy);
            lruIndex_[dummy] = lruList_.begin();
            // 填地址映射
            addressMap_[dummy] = addr;
        }
    }

    ~MemoryPool() { delete[] pool_; }

    MemoryPool(const MemoryPool&)            = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    /* ---------------- 对外接口 ---------------- */
    Status NewBlock(const std::string& blockId) {
        if (addressMap_.count(blockId)) return Status::DuplicateKey();
        if (lruList_.empty()) return Status::Error();
        char* addr = evictLRU();
        addressMap_[blockId] = addr;
        return Status::OK();
    }

    bool LookupBlock(const std::string& blockId) const {
        return availableBlocks_.count(blockId);
    }

    char* GetAddress(const std::string& blockId) const {
        auto it = addressMap_.find(blockId);
        return it == addressMap_.end() ? nullptr : it->second;
    }

    Status CommitBlock(const std::string& blockId, bool success) {
        if (success) {
            availableBlocks_.insert(blockId);
            touchUnsafe(blockId);
        } else {
            // availableBlocks_.erase(blockId); // 这句大概不需要？
            auto it = addressMap_.find(blockId);
            char* addr = it->second;
            int32_t offset = static_cast<uint32_t>(addr - pool_);
            std::string dummy = "__slot_" + std::to_string(offset / blockSize_);
            addressMap_.erase(blockId);

            auto lit = lruIndex_.find(blockId);
            if (lit != lruIndex_.end()) {
                lruList_.erase(lit->second);
                lruIndex_.erase(lit);
            }
            lruList_.push_back(dummy);
            lruIndex_[dummy] = std::prev(lruList_.end());
            addressMap_[dummy] = addr;
        }
        return Status::OK();
    }

    char* GetFirstAddr() {
        return pool_;
    }

private:
    /* ---------------- 内部数据 ---------------- */
    char* pool_ = nullptr;
    uint32_t capacity_;
    uint32_t blockSize_;
    uint32_t slotNum_;

    std::unordered_map<std::string, char*> addressMap_;
    std::set<std::string> availableBlocks_;

    using ListType = std::list<std::string>;
    ListType lruList_;
    std::unordered_map<std::string, ListType::iterator> lruIndex_;

    /* ---------------- 工具函数 ---------------- */
    // 把 blockId 移到 LRU 头
    void touchUnsafe(const std::string& blockId) {
        auto it = lruIndex_.find(blockId);
        if (it != lruIndex_.end()) {
            lruList_.splice(lruList_.begin(), lruList_, it->second);
        } else {
            lruList_.push_front(blockId);
            lruIndex_[blockId] = lruList_.begin();
        }
    }

    // 踢最久未使用块
    char* evictLRU() {
        const std::string& victim = lruList_.back();
        // 真数据块才清可用集合
        if (victim.rfind("__slot_", 0) != 0) {
            availableBlocks_.erase(victim);
        }
        char* addr = addressMap_[victim];
        addressMap_.erase(victim);
        lruIndex_.erase(victim);
        lruList_.pop_back();
        return addr;
    }
};

} // namespace UC
#endif