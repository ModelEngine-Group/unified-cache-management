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
#include "device/idevice.h"
#include <stdexcept>
#include <iostream>
#include <memory>
#include "logger/logger.h"

namespace UC {

class MemoryPool {

    std::string DUMMY_SLOT_PREFIX{"__slot_"};
    using Device = std::unique_ptr<IDevice>;
public:
    // MemoryPool(int32_t deviceId, size_t capacity, size_t blockSize) {
    //     capacity_ = capacity;
    //     blockSize_ = blockSize;
    //     deviceId_ = deviceId;
        // device_ = DeviceFactory::Make(deviceId, blockSize, static_cast<int>(capacity / blockSize)); // 大小是内存池的总容量大小
        // if (!device_) {
        //     throw std::runtime_error("MemoryPool::MemoryPool() failed due to failure to initialize device");
        // }
        // Status success = device_->Setup();
        // if (!success.Success()) {
        //     throw std::runtime_error("MemoryPool::MemoryPool() failed due to failure to setup device");
        // }
        // pool_ = device_->GetBuffer(capacity_);

        // if (!pool_) {
        //     throw std::bad_alloc();
        // }
        // size_t slotNum = capacity / blockSize;
        // for (size_t i = 0; i < slotNum; ++i) {
        //     // 将所有槽位都预先占好，插入LRU队列中。
        //     std::string dummy = DUMMY_SLOT_PREFIX + std::to_string(i);
        //     // std::shared_ptr<std::byte> addr = pool_ + i * blockSize_;
        //     size_t offset = i * blockSize_;
        //     lruList_.push_front(dummy);
        //     lruIndex_[dummy] = lruList_.begin();
        //     // offsetMap_[dummy] = addr;
        //     offsetMap_[dummy] = offset;
        // }

    Status Setup(int32_t deviceId, size_t capacity, size_t blockSize) {
        capacity_ = capacity;
        blockSize_ = blockSize;
        device_ = DeviceFactory::Make(deviceId, blockSize, static_cast<int>(capacity / blockSize));
        if (!device_) {
            UC_ERROR("MemoryPool: failed to create device");
            return Status::Error();
        }
        Status status = device_->Setup();
        if (!status.Success()) {
            UC_ERROR("MemoryPool: failed to set up device");
            return Status::Error();
        }
        pool_ = device_->GetBuffer(capacity_);
        if (!pool_) {
            UC_ERROR("MemoryPool: failed to get pool memory space");
            return Status::Error();
        }

        size_t slotNum = capacity_ / blockSize_;
        for (size_t i = 0; i < slotNum; ++i) {
            std::string dummy = DUMMY_SLOT_PREFIX + std::to_string(i);
            size_t offset = i * blockSize_;
            lruList_.push_front(dummy);
            lruIndex_[dummy] = lruList_.begin();
            offsetMap_[dummy] = offset;
        return Status::OK();

    }

    // MemoryPool(const MemoryPool&) = delete;
    // MemoryPool& operator=(const MemoryPool&) = delete;

    Status NewBlock(const std::string& blockId) {
        if (offsetMap_.count(blockId)) {
            return Status::DuplicateKey();
        }
        if (lruList_.empty()) {
            // 所有空间里的块都正在写，那么就不能够分配
            return Status::Error();
        }
        size_t offset = LRUEvictOne();
        offsetMap_[blockId] = offset;
        return Status::OK();
    }

    bool LookupBlock(const std::string& blockId) const {
        return availableBlocks_.count(blockId);
    }

    bool GetOffset(const std::string& blockId, size_t* offset) const {
        auto it = offsetMap_.find(blockId);
        if (it == offsetMap_.end()) {
            return false;
        }
        *offset = it->second;
        return true;
    }

    Status CommitBlock(const std::string& blockId, bool success) {
        if (success) {
            availableBlocks_.insert(blockId);
            touchUnsafe(blockId);
        } else {
            resetSpaceOfBlock(blockId);
        }
        return Status::OK();
    }

    std::shared_ptr<std::byte> GetStartAddr() const {
        return pool_;
    }

private:
    std::shared_ptr<std::byte> pool_ = nullptr;
    Device device_ = nullptr;
    size_t capacity_;
    size_t blockSize_;

    std::unordered_map<std::string, size_t> offsetMap_;
    std::set<std::string> availableBlocks_;

    using ListType = std::list<std::string>;
    ListType lruList_;
    std::unordered_map<std::string, ListType::iterator> lruIndex_;

    void touchUnsafe(const std::string& blockId) {
        auto it = lruIndex_.find(blockId);
        if (it != lruIndex_.end()) {
            lruList_.splice(lruList_.begin(), lruList_, it->second);
        }
        else {
            lruList_.push_front(blockId); // 访问一次，该块就是最近使用了的，所以放到LRU队列的头部。这就是一般LRU的逻辑
            lruIndex_[blockId] = lruList_.begin();
        }
    }

    size_t LRUEvictOne() {
        const std::string& victim = lruList_.back();
        // 真实数据块，才从availableBlocks_中删掉
        if (victim.rfind(DUMMY_SLOT_PREFIX, 0) != 0) {
            availableBlocks_.erase(victim);
        }
        size_t offset = offsetMap_[victim];
        offsetMap_.erase(victim);
        lruIndex_.erase(victim);
        lruList_.pop_back();
        return offset;
    }

    void resetSpaceOfBlock(const std::string& blockId) {
        // availableBlocks_.erase(blockId); // 这句大概不需要？
        auto it = offsetMap_.find(blockId);
        // int32_t offset = static_cast<size_t>(addr - pool_);
        size_t offset = it->second;
        std::string dummy = DUMMY_SLOT_PREFIX + std::to_string(offset / blockSize_);
        offsetMap_.erase(blockId);

        auto lit = lruIndex_.find(blockId);
        if (lit != lruIndex_.end()) {
            lruList_.erase(lit->second);
            lruIndex_.erase(lit);
        }
        lruList_.push_back(dummy); // 将一个块commit false后，回收之前分配的内存，并且要将其放到LRU队列的尾部（下次可以写的时候，要马上就写。因为该块的优先级高于已经写了的块）
        lruIndex_[dummy] = std::prev(lruList_.end());
        offsetMap_[dummy] = offset;
    }
};

} // namespace UC
#endif