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
#include "thread_pool/thread_pool.h"

namespace UCM {

Status CacheManager::Setup(const size_t capacity, const size_t cacheSize,
                           const int deviceId, const size_t ioSize)
{
    if (capacity == 0 || cacheSize == 0) {
        UCM_ERROR("Invalid param({},{}) for cache.", capacity, cacheSize);
        return Status::InvalidParam();
    }
    auto status = this->_lruCache.Init(capacity, cacheSize);
    if (status.Failure()) {
        UCM_ERROR("Failed({}) to init LRUCache.", status);
        return status;
    }
    this->_cacheSize = cacheSize;
    this->_deviceId = deviceId;
    this->_ioSize = ioSize;
    return Status::OK();
}

Status CacheManager::Cache2Device(const std::string& blockId, const uintptr_t dst, const size_t length, const size_t offset)
{
    void* buffer = nullptr;
    auto status = this->_lruCache.Find(blockId, &buffer);
    if (buffer == nullptr) {
        return Status::NotFound();
    }

    auto cacheId = this->_cacheIdSeed;
    this->_cacheStates[cacheId]->pendingOps.fetch_add(1);
    ThreadPool::Instance().Submit([this, cacheId, dst, length, buffer, offset]() {
        auto status = Status::OK();
        std::unique_ptr<IDevice> device = nullptr;
        if (this->_deviceId >= 0){
            if (!(device = Device::Make(this->_deviceId, this->_ioSize))){
                UCM_ERROR("Failed to make device.");
                return;
            }
            if ((status = device->Setup()).Failure()){
                UCM_ERROR("Failed({}) to setup device.", status);
                return;
            }
        }
        status = device->H2DAsync((void*)dst, length, (char*)buffer + offset, length);
        if (status.Failure()) {
            UCM_ERROR("Failed({}) to load data from cache to device.", status);
            return;
        }
        device->WaitFinish();
        this->_lruCache.FindCommit(buffer);
        this->_cacheStates[cacheId]->pendingOps.fetch_sub(1);
        this->_cacheStates[cacheId]->cv.notify_all();
    });
    return status;
}

Status CacheManager::AllocBuffers(std::list<std::string> blockIds, std::list<uintptr_t>& buffers)
{
    for (auto& blockId : blockIds) {
        void* buffer = nullptr;
        auto status = this->_lruCache.Alloc(blockId, &buffer);
        while (status == Status::Busy()) {
            status = this->_lruCache.Alloc(blockId, &buffer);
        }
        if (status.Failure()) {
            UCM_ERROR("Failed({}) to alloc buffer on dram", status);
            return status;
        }
        buffers.push_back(reinterpret_cast<uintptr_t>(buffer));
    }
    return Status::OK();
}

Status CacheManager::CommitBuffers(std::list<uintptr_t>& buffers) {
    for (auto it = buffers.begin(); it != buffers.end(); ++it) {
        void* buffer = reinterpret_cast<void*>(*it);
        this->_lruCache.AllocCommit(buffer);
    }
    buffers.clear();
    return Status::OK();
}

Status CacheManager::Wait(const size_t cacheId)
{
    std::unique_lock<std::mutex> lk(this->_cacheStates[cacheId]->mtx);
    this->_cacheStates[cacheId]->cv.wait(lk, [this, cacheId] {
        return this->_cacheStates[cacheId]->pendingOps == 0;
    });
    this->_cacheStates.erase(cacheId);
    return Status::OK();
}

} // namespace UCM
