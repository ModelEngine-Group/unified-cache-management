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
#ifndef ATB_KVCACHE_PREFETCH_H
#define ATB_KVCACHE_PREFETCH_H
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <omp.h>
#include <pybind11/numpy.h>
#include <queue>
#include <sstream>
#include <stdarg.h>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "ucmstore_v1.h"

namespace py = pybind11;
namespace ucmprefetch {
using BlockId = std::array<std::byte, 16>; /* 16-byte block hash */

class __attribute__((visibility("hidden"))) UCMKVPrefetch {
private:
    UC::StoreV1* mStoreV1_{nullptr};
    std::map<int32_t, UC::Detail::TaskHandle> mTaskHandle;
    std::mutex mTaskMu_;
    std::condition_variable mTaskCv_;
    std::queue<int32_t> mWaitQueue_;
    std::thread mAsyncWaiter_;
    std::atomic<bool> mStop_{false};
    int32_t mTaskID = 0;
    uint32_t defaultLayerNum = 78;

    /* 后台等待线程：阻塞等待 load 完成，随后自动释放临时 handle / 缓存。 */
    void AsyncWaitLoop()
    {
        for (;;) {
            int32_t taskID = -1;
            {
                std::unique_lock<std::mutex> lock(mTaskMu_);
                mTaskCv_.wait(lock, [this] { return mStop_.load() || !mWaitQueue_.empty(); });
                if (mStop_.load() && mWaitQueue_.empty()) { return; }
                taskID = mWaitQueue_.front();
                mWaitQueue_.pop();
            }
            UCMWaitInternal(taskID);
        }
    }

    void UCMWaitInternal(int32_t taskID)
    {
        UC::Detail::TaskHandle taskHandle = 0;
        {
            std::unique_lock<std::mutex> lock(mTaskMu_);
            auto iter = mTaskHandle.find(taskID);
            if (iter == mTaskHandle.end()) { return; /* 已被清理，无需再等待 */ }
            taskHandle = iter->second;
        }
        /* 无论等待成功/失败/异常，都保证该任务的 handle 条目被清理，避免泄漏。 */
        try {
            if (mStoreV1_ == nullptr) {
                std::cerr << "[UCMKVPrefetch] storeV1 is null, task " << taskID << " cannot wait"
                          << std::endl;
            } else {
                auto status = mStoreV1_->WaitSSDToHost(taskHandle);
                if (status.Failure()) {
                    std::cerr << "[UCMKVPrefetch] wait task " << taskID
                              << " failed: " << status.ToString() << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[UCMKVPrefetch] wait task " << taskID << " exception: " << e.what()
                      << std::endl;
        } catch (...) {
            std::cerr << "[UCMKVPrefetch] wait task " << taskID << " unknown exception"
                      << std::endl;
        }
        {
            std::unique_lock<std::mutex> lock(mTaskMu_);
            mTaskHandle.erase(taskID);
        }
        mTaskCv_.notify_all();
    }

public:
    UCMKVPrefetch() { mAsyncWaiter_ = std::thread(&UCMKVPrefetch::AsyncWaitLoop, this); }

    ~UCMKVPrefetch()
    {
        {
            std::unique_lock<std::mutex> lock(mTaskMu_);
            mStop_.store(true);
        }
        mTaskCv_.notify_all();
        if (mAsyncWaiter_.joinable()) { mAsyncWaiter_.join(); }
    }

    void StoreInit(uintptr_t storePtr)
    {
        mStoreV1_ = reinterpret_cast<UC::StoreV1*>(storePtr);
        if (mStoreV1_ == nullptr) { throw std::runtime_error("storeV1 is null"); }
    }
    int32_t UCMLoadSSD(const std::vector<UC::Detail::BlockId>& blockIds)
    {
        if (mStoreV1_ == nullptr) { throw std::runtime_error("storeV1 is null"); }

        if (blockIds.empty()) { throw std::runtime_error("blockIds is empty"); }
        UC::Detail::TaskDesc task;
        task.reserve(blockIds.size() * defaultLayerNum);

        for (const auto& blockId : blockIds) {
            for (uint32_t layerID = 0; layerID < defaultLayerNum; layerID++) {
                task.emplace_back(UC::Detail::Shard{blockId, layerID});
            }
        }

        auto loadHandle = mStoreV1_->LoadSSDToHost(std::move(task));
        if (!loadHandle) { throw std::runtime_error(loadHandle.Error().ToString()); }
        mTaskID += 1;
        mTaskHandle[mTaskID] = loadHandle.Value();
        return mTaskID;
    }

    /* 显式阻塞等待（可选）：等待后台线程完成该任务的等待与清理。 */
    void UCMWait(int32_t taskID)
    {
        std::unique_lock<std::mutex> lock(mTaskMu_);
        mTaskCv_.wait(lock,
                      [this, taskID] { return mTaskHandle.find(taskID) == mTaskHandle.end(); });
    }

    std::vector<uint8_t> UCMLookup(const std::vector<UC::Detail::BlockId>& blockIds)
    {
        if (mStoreV1_ == nullptr) { throw std::runtime_error("storeV1 is null"); }

        if (blockIds.empty()) { throw std::runtime_error("blockIds is empty"); }

        auto lookupResult = mStoreV1_->Lookup(blockIds.data(), blockIds.size());
        if (!lookupResult) { throw std::runtime_error(lookupResult.Error().ToString()); }

        auto result = lookupResult.Value();
        if (result.size() != blockIds.size()) {
            throw std::runtime_error("lookup result size does not match blockIds size");
        }

        return result;
    }
};

}  // namespace ucmprefetch

#endif