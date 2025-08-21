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
#ifndef UCM_LOCAL_STORE_CACHE_MANAGER_H
#define UCM_LOCAL_STORE_CACHE_MANAGER_H

#include "lrucache/lrucache.h"
#include <mutex>
#include <list>
#include <future>
#include <condition_variable>
#include <functional>
#include <memory>
#include "status/status.h"
#include "device/idevice.h"

namespace UCM {

struct CacheState {
    std::atomic<size_t> pendingOps{0};
    std::mutex mtx;
    std::condition_variable cv;
};

struct FindResult {
    Status status;
    std::string blockId;
    void* buffer;
};

class CacheManager {
public:
    Status Setup(const size_t capacity, const size_t cacheSize, const int deviceId, const size_t ioSize);
    std::list<bool> LookupBatch(const std::list<std::string>& blockIdList);
    size_t SubmitWrite(const std::list<std::string>& blockIdList, const std::list<uintptr_t> srcList);
    size_t SubmitRead(const std::list<std::string>& blockIdList, const std::list<uintptr_t> dstList,
        const std::list<size_t> lengthList, const std::list<size_t> offsetList);
    Status Wait(const size_t cacheId);
private:
    Status WriteCache(const std::string& blockId, const uintptr_t src);
    Status ReadCache(const std::string& blockId, void* buffer, const uintptr_t dst, const size_t length, const size_t offset);
private:
    LRUCache _lruCache;
    size_t _cacheSize;
    size_t _deviceId;
    size_t _ioSize;
    std::mutex _mtx;
    std::atomic<size_t> _cacheIdSeed{0};
    std::unordered_map<size_t, std::shared_ptr<CacheState>> _cacheStates;
    std::unordered_map<std::string, void*> _blockIdToCache;
    std::vector<std::future<FindResult>> _futures;
};

} // namespace UCM

#endif
