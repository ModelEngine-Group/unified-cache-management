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

namespace UC {

class MemoryPool {
    using Device = std::unique_ptr<IDevice>;
public:
    MemoryPool(int32_t deviceId, size_t capacity, size_t blockSize) {
        capacity_ = capacity;
        blockSize_ = blockSize;
        // pool_ = new char[capacity];
        device_ = DeviceFactory::Make(deviceId, blockSize, static_cast<int>(capacity / blockSize)); // 大小是内存池的总容量大小
        if (!device_) {
            throw std::runtime_error("MemoryPool::MemoryPool() failed due to failure to initialize device");
        }
        Status success = device_->Setup();
        if (!success.Success()) {
            throw std::runtime_error("MemoryPool::MemoryPool() failed due to failure to setup device");
        }
        pool_ = static_cast<char*>(device_->GetBuffer(capacity_));

        if (!pool_) {
            throw std::bad_alloc();
        }
        size_t slotNum = capacity / blockSize;
        for (size_t i = 0; i < slotNum; ++i) {
            // 将所有槽位都预先占好，插入LRU队列中。
            std::string dummy = "__slot_" + std::to_string(i);
            char* addr = pool_ + i * blockSize_;
            lruList_.push_front(dummy);
            lruIndex_[dummy] = lruList_.begin();
            addressMap_[dummy] = addr;
        }
    }

    ~MemoryPool() {
        delete[] pool_;
    }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    Status NewBlock(const std::string& blockId) {
        if (addressMap_.count(blockId)) {
            return Status::DuplicateKey();
        }
        if (lruList_.empty()) {
            // 所有空间里的块都正在写，那么就不能够分配
            return Status::Error();
        }
        char* addr = LRUEvictOne();
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
            resetSpaceOfBlock(blockId);
        }
        return Status::OK();
    }

    // 单元测试用，外部应该用不到
    char* GetFirstAddr() {
        return pool_;
    }

private:
    char* pool_ = nullptr;
    Device device_ = nullptr;
    size_t capacity_;
    size_t blockSize_;

    std::unordered_map<std::string, char*> addressMap_;
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

    char* LRUEvictOne() {
        const std::string& victim = lruList_.back();
        // 真实数据块，才从availableBlocks_中删掉
        if (victim.rfind("__slot_", 0) != 0) {
            availableBlocks_.erase(victim);
        }
        char* addr = addressMap_[victim];
        addressMap_.erase(victim);
        lruIndex_.erase(victim);
        lruList_.pop_back();
        return addr;
    }

    void resetSpaceOfBlock(const std::string& blockId) {
        // availableBlocks_.erase(blockId); // 这句大概不需要？
        auto it = addressMap_.find(blockId);
        char* addr = it->second;
        int32_t offset = static_cast<size_t>(addr - pool_);
        std::string dummy = "__slot_" + std::to_string(offset / blockSize_);
        addressMap_.erase(blockId);

        auto lit = lruIndex_.find(blockId);
        if (lit != lruIndex_.end()) {
            lruList_.erase(lit->second);
            lruIndex_.erase(lit);
        }
        lruList_.push_back(dummy); // 将一个块commit false后，回收之前分配的内存，并且要将其放到LRU队列的尾部（下次可以写的时候，要马上就写。因为该块的优先级高于已经写了的块）
        lruIndex_[dummy] = std::prev(lruList_.end());
        addressMap_[dummy] = addr;
    }
};

} // namespace UC
#endif