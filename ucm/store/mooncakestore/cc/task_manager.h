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
#ifndef UNIFIEDCACHE_MOONCAKE_STORE_CC_TASK_MANAGER_H
#define UNIFIEDCACHE_MOONCAKE_STORE_CC_TASK_MANAGER_H

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "host_buffer_pool.h"
#include "status/status.h"
#include "thread/thread_pool.h"
#include "trans_task.h"
#include "type/types.h"

namespace UC::MooncakeStore {

class TaskManager {
public:
    using LoadProcessFn = std::function<void(Detail::TaskHandle, TransTask&, HostBufferPool&)>;
    using DumpProcessFn = std::function<void(Detail::TaskHandle, TransTask&)>;

    TaskManager();
    ~TaskManager();

    Status Setup(uint32_t loadWorkerNum, uint32_t dumpWorkerNum, HostBufferPool& bufPool,
                 LoadProcessFn loadFn, DumpProcessFn dumpFn);
    void Close();

    Expected<Detail::TaskHandle> SubmitLoad(TransTask task);
    Expected<Detail::TaskHandle> SubmitDump(TransTask task);

    Expected<bool> Check(Detail::TaskHandle handle);
    Status Wait(Detail::TaskHandle handle);

    std::shared_ptr<TaskState> GetState(Detail::TaskHandle handle);

private:
    Expected<Detail::TaskHandle> Submit(TransTask task, bool isLoad);

    LoadProcessFn loadFn_;
    DumpProcessFn dumpFn_;
    HostBufferPool* bufPool_{nullptr};

    std::unique_ptr<ThreadPool<PendingItem>> loadPool_;
    std::unique_ptr<ThreadPool<PendingItem>> dumpPool_;

    std::mutex taskMtx_;
    std::unordered_map<Detail::TaskHandle, std::shared_ptr<TaskState>> tasks_;

    std::atomic<Detail::TaskHandle> nextTaskId_{1};
    std::atomic<bool> closed_{false};
};

}  // namespace UC::MooncakeStore

#endif
