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
#include "cache/cache_manager.h"
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
    auto status = this->_lruCache.Initialize(capacity, cacheSize);
    if (status.Failure()) {
        UCM_ERROR("Failed({}) to init LRUCache.", status);
        return status;
    }
    this->_cacheSize = cacheSize;
    this->_deviceId = deviceId;
    this->_ioSize = ioSize;
    return Status::OK();
}

std::list<bool> CacheManager::LookupBatch(const std::list<std::string>& blockIdList)
{
    std::list<bool> founds;
    for (auto& blockId : blockIdList) {
        this->_futures.push_back(ThreadPool::Instance().Submit([this, blockId]() -> FindResult {
            void* buffer = nullptr;
            auto status = this->_lruCache.Find(blockId, &buffer);
            return FindResult{status, blockId, buffer};
        }));
    }
    for (auto& future : this->_futures) {
        FindResult result = future.get();
        if (result.status == Status::OK() && result.buffer != nullptr) {
            this->_blockIdToCache[result.blockId] = result.buffer;
            founds.push_back(true);
        } else {
            founds.push_back(false);
        }
    }
    this->_futures.clear();
    return founds;
}

Status CacheManager::WriteCache(const std::string& blockId, const uintptr_t src)
{
    void* buffer = nullptr;
    auto status = this->_lruCache.Insert(blockId, &buffer);
    if (buffer == nullptr) {
        UCM_ERROR("Failed({}) to alloc buffer on dram.", status);
        return status;
    }
    memcpy(buffer, reinterpret_cast<const void*>(src), this->_cacheSize);
    this->_lruCache.Done(buffer);
    return Status::OK();
}

Status CacheManager::ReadCache(const std::string& blockId, void* buffer, const uintptr_t dst, const size_t length, const size_t offset)
{
    auto status = Status::OK();
    std::unique_ptr<IDevice> device = nullptr;
    if (this->_deviceId >= 0){
        if (!(device = Device::Make(this->_deviceId, this->_ioSize))){
            UCM_ERROR("Failed to make device.");
            return Status::InvalidParam();
        }
        if ((status = device->Setup()).Failure()){
            UCM_ERROR("Failed({}) to setup device.", status);
            return Status::InvalidParam();
        }
    }
    status = device->H2DAsync((void*)dst, length, (char*)buffer + offset, length);
    if (status.Failure()) {
        UCM_ERROR("Failed({}) to load data from cache to device.", status);
        return status;
    }
    device->WaitFinish();
    this->_lruCache.Done(buffer);
    {
        std::lock_guard<std::mutex> lk(this->_mtx);
        this->_blockIdToCache.erase(blockId);
    }
    return status;
}

size_t CacheManager::SubmitWrite(const std::list<std::string>& blockIdList, const std::list<uintptr_t> srcList)
{
    size_t cacheId = this->_cacheIdSeed.fetch_add(1) + 1;
    std::shared_ptr<CacheState> cacheState = std::make_shared<CacheState>();
    cacheState->pendingOps.store(blockIdList.size());
    {
        std::unique_lock<std::mutex> lk(this->_mtx);
        this->_cacheStates[cacheId] = cacheState;
    }

    auto itBlock = blockIdList.begin();
    auto itSrc   = srcList.begin();
    for (; itBlock != blockIdList.end() && itSrc != srcList.end(); ++itBlock, ++itSrc) {
        auto blockId = *itBlock;
        auto src = *itSrc;
        ThreadPool::Instance().Submit([this, blockId, src, cacheState]() {
            auto status = this->WriteCache(blockId, src);
            if (status.Failure()) {
                UCM_ERROR("Failed({}) to write cache({}).", status, blockId);
            }
            cacheState->pendingOps.fetch_sub(1);
            cacheState->cv.notify_all();
        });
    }
    return cacheId;
}

size_t CacheManager::SubmitRead(const std::list<std::string>& blockIdList, const std::list<uintptr_t> dstList,
        const std::list<size_t> lengthList, const std::list<size_t> offsetList)
{
    size_t cacheId = this->_cacheIdSeed.fetch_add(1) + 1;
    std::shared_ptr<CacheState> cacheState = std::make_shared<CacheState>();
    cacheState->pendingOps.store(blockIdList.size());
    {
        std::unique_lock<std::mutex> lk(this->_mtx);
        this->_cacheStates[cacheId] = cacheState;
    }

    auto itBlock  = blockIdList.begin();
    auto itDst    = dstList.begin();
    auto itLen    = lengthList.begin();
    auto itOffset = offsetList.begin();
    for (; itBlock != blockIdList.end(); ++itBlock, ++itDst, ++itLen, ++itOffset) {
        auto blockId = *itBlock;
        auto dst     = *itDst;
        auto length  = *itLen;
        auto offset  = *itOffset;
        ThreadPool::Instance().Submit([this, blockId, dst, length, offset, cacheState]() {
            void* buffer = nullptr;
            auto it = this->_blockIdToCache.find(blockId);
            if (it == this->_blockIdToCache.end()) {
                UCM_ERROR("BlockId({}) not found in cache.", blockId);
                cacheState->pendingOps.fetch_sub(1);
                cacheState->cv.notify_all();
                return;
            }
            buffer = it->second;
            auto status = this->ReadCache(blockId, buffer, dst, length, offset);
            if (status.Failure()) {
                UCM_ERROR("Failed({}) to read cache({}).", status, blockId);
            }
            cacheState->pendingOps.fetch_sub(1);
            cacheState->cv.notify_all();
        });
    }
    return cacheId;
}

Status CacheManager::Wait(const size_t cacheId)
{
    auto it = this->_cacheStates.find(cacheId);
    if (it == this->_cacheStates.end()) {
        return Status::NotFound();
    }
    auto state = it->second;
    std::unique_lock<std::mutex> lk(state->mtx);
    state->cv.wait(lk, [state] { return state->pendingOps == 0; });
    this->_cacheStates.erase(it);
    return Status::OK();
}

} // namespace UCM
