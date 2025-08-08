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
#include "cache_manager.h"
#include "file/file.h"
#include "logger/logger.h"
#include "device/device.h"

namespace UCM {

Status CacheManager::Setup(const size_t capacity, const size_t cacheSize)
{
    if (capacity == 0 || cacheSize == 0) {
        UCM_ERROR("Invalid param({},{}) for cache.", capacity, cacheSize);
        return Status::InvalidParam();
    }
    auto status = this->_lruCache.Initialize(capacity, cacheSize);
    if (status.Failure()) {
        UCM_ERROR("Failed({}) to init LRUCache.", status);
        return status;
    }
    return Status::OK();
}

Status CacheManager::ReadCache(TsfTask& task)
{
    if (task.buffer != nullptr) {
        return Status::OK();
    }
    auto status = this->_lruCache.Find(task.blockId, task.buffer);
    if (status.Failure() && (status != Status::NotFound()) && (status != Status::Empty())) {
        UCM_ERROR("Failed({}) to read cache.", status);
        return status;
    }
    return Status::OK();
}

Status CacheManager::AllocBuffers(const std::list<std::string>& blockIds, std::list<std::string>& uniqueBlockIds,
                                  std::list<uintptr_t>& buffers)
{
    for (auto& blockId : blockIds) {
        void* buffer = nullptr;
        auto status = this->_lruCache.Insert(blockId, buffer);
        while (status == Status::Busy()) {
            status = this->_lruCache.Insert(blockId, buffer);
        }
        if (status == Status::Exist()) {
            continue;
        }
        if (status.Failure()) {
            UCM_ERROR("Failed({}) to alloc buffer on dram", status);
            return status;
        }
        uniqueBlockIds.push_back(blockId);
        buffers.push_back(reinterpret_cast<uintptr_t>(buffer));
    }
    return Status::OK();
}

Status CacheManager::CommitBuffers(std::list<uintptr_t>& buffers) {
    for (auto it = buffers.begin(); it != buffers.end(); ++it) {
        void* buffer = reinterpret_cast<void*>(*it);
        this->_lruCache.Done(buffer);
    }
    buffers.clear();
    return Status::OK();
}

} // namespace UCM
