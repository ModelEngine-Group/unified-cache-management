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

namespace UC {

#pragma once
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include "status/status.h"

class MemoryPool {
public:
    MemoryPool(uint32_t capacity, uint32_t blockSize) : capacity_(capacity), head_(0), blockSize_(blockSize) {
        pool_ = new char[capacity];
        if (!pool_) {
            throw std::bad_alloc();
        }
    }

    ~MemoryPool() {
        delete [] pool_;
    }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    void Reset() {
        head_ = 0;
    }

    uint32_t GetNextAvailableOffset() const { return head_; }

    Status NewBlock(std::string blockId) {
        if (head_ >= capacity_) {
            return Status::Error(); // 下一版本再实现GC逻辑，目前先忽略吧
        }
        auto it = addressMap_.find(blockId);
        if (it != addressMap_.end()) {
            // duplicate key
            return Status::OK();
        }
        // addressMap_里目前还没有这个blockId，即将进行分配
        addressMap_[blockId] = pool_ + head_;
        head_ = head_ + blockSize_;
        return Status::OK();
    }

    bool LookupBlock(std::string blockId) {
        return availableBlocks_.find(blockId) != availableBlocks_.end();
    }

    char* GetAddress(std::string blockId) {
        if (addressMap_.find(blockId) == addressMap_.end()) {
            return nullptr;
        }
        return addressMap_[blockId];
    }

    Status CommitBlock(std::string blockId, bool success) {
        if (success) {
            availableBlocks_.insert(blockId)
        }
        else {
            availableBlocks_.erase(blockId);
        }
        return Status::OK();
    }

private:
    char* pool_ = nullptr;
    uint32_t capacity_;
    uint32_t head_;
    uint32_t blockSize_;
    std::unordered_map<std::string, char*> addressMap_;
    std::set<std::string> availableBlocks_;
};

} // namespace UC

#endif
