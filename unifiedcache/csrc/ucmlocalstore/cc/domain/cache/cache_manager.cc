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

Status CacheManager::ReadCache(const std::string& blockId, void* buffer, const uintptr_t dst,
                               const size_t length, const size_t offset)
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
    return status;
}

std::list<bool> CacheManager::Cache2Device(std::shared_ptr<CacheState> cacheState, const std::list<std::string>& blockIdList,
                                           const std::list<uintptr_t>& dstList, const std::list<size_t>& lengthList,
                                           const std::list<size_t>& offsetList)
{
    std::list<bool> founds;
    std::vector<std::future<FindResult>> futures;
    auto itBlock = blockIdList.begin();
    auto itDst    = dstList.begin();
    auto itLen    = lengthList.begin();
    auto itOffset = offsetList.begin();
    for (; itBlock != blockIdList.end(); ++itBlock, ++itDst, ++itLen, ++itOffset) {
        futures.push_back(ThreadPool::Instance().Submit(
            [this, blockId = *itBlock, dst = *itDst, len = *itLen, offset = *itOffset]() -> FindResult {
                void* buffer = nullptr;
                auto status = this->_lruCache.Find(blockId, &buffer);
                return FindResult{status, blockId, dst, offset, len, buffer};
            }
        ));
    }
    for (auto& future : futures) {
        FindResult result = future.get();
        if (result.status == Status::OK() && result.buffer != nullptr) {
            ThreadPool::Instance().Submit([this, result, cacheState]() {
                auto status = this->ReadCache(result.blockId, result.buffer, result.dst,
                    result.length, result.offset);
                if (status.Failure()) {
                    UCM_ERROR("Failed({}) to read cache({}).", status, result.blockId);
                }
                cacheState->pendingOps.fetch_sub(1);
                cacheState->cv.notify_all();
            });
            founds.push_back(true);
        } else {
            cacheState->pendingOps.fetch_sub(1);
            founds.push_back(false);
        }
    }
    return founds;
}

Status CacheManager::AllocBuffers(const std::list<std::string>& blockIds, std::list<uintptr_t>& buffers)
{
    for (auto& blockId : blockIds) {
        void* buffer = nullptr;
        auto status = this->_lruCache.Insert(blockId, &buffer);
        while (status == Status::Busy()) {
            status = this->_lruCache.Insert(blockId, &buffer);
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
        this->_lruCache.Done(buffer);
    }
    buffers.clear();
    return Status::OK();
}

Status CacheManager::Wait(const size_t cacheId, size_t& taskId)
{
    auto it = this->_cacheStates.find(cacheId);
    if (it == this->_cacheStates.end()) {
        return Status::NotFound();
    }
    auto state = it->second;
    std::unique_lock<std::mutex> lk(state->mtx);
    state->cv.wait(lk, [state] { return state->pendingOps == 0; });
    this->_cacheStates.erase(it);
    taskId = state->taskId;
    return Status::OK();
}

} // namespace UCM
