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
#include <condition_variable>
#include <functional>
#include <memory>
#include "status/status.h"
#include "device/idevice.h"

namespace UCM {

struct TaskState {
    std::atomic<int> pendingOps{0};
    std::mutex mtx;
    std::condition_variable cv;
};

class CacheManager {
public:
    Status Setup(const size_t capacity, const size_t cacheSize, const int deviceId, const size_t ioSize);
    size_t GetCacheId() {
        std::unique_lock<std::mutex> lck(this->_mtx);
        this->_cacheIdSeed++;
        this->_cacheStates.emplace(this->_cacheIdSeed, std::make_shared<TaskState>());
        return this->_cacheIdSeed;
    }
    Status Cache2Device(const std::string& blockId, const uintptr_t dst, const size_t length, const size_t offset);
    Status AllocBuffers(std::list<std::string> blockIds, std::list<uintptr_t>& buffers);
    Status CommitBuffers(std::list<uintptr_t>& buffers);
    Status Wait(const size_t cacheId);
    void AddTaskToCache(size_t taskId, size_t cacheId) {
        this->_cacheToTask[taskId] = cacheId;
    }
    bool FindCacheByTask(const size_t taskId, size_t& cacheId) {
        auto it = this->_cacheToTask.find(taskId);
        if (it != this->_cacheToTask.end()) {
            cacheId = it->second;
            return true;
        } else {
            return false;
        }
    }
private:
    LRUCache _lruCache;
    size_t _cacheSize;
    std::unordered_map<size_t, std::shared_ptr<TaskState>> _cacheStates;
    std::unordered_map<size_t, size_t> _cacheToTask;
    std::mutex _mtx;
    size_t _cacheIdSeed{0};
    size_t _deviceId;
    size_t _ioSize;
};

} // namespace UCM

#endif
