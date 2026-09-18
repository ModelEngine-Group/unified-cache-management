/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_YUANRONG_STORE_CC_TASK_MANAGER_H
#define UNIFIEDCACHE_YUANRONG_STORE_CC_TASK_MANAGER_H

#include "dump_queue.h"
#include "load_queue.h"
#include "template/task_wrapper.h"
#include "trans_task.h"

namespace UC::YuanRongStore {

class TaskManager : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
    LoadQueue loadQueue_;
    DumpQueue dumpQueue_;

public:
    Status Setup(const Config& config, std::shared_ptr<datasystem::HeteroClient> heteroClient,
                 std::shared_ptr<datasystem::KVClient> kvClient)
    {
        timeoutMs_ = config.timeoutMs;
        auto s = loadQueue_.Setup(config, &failureSet_, heteroClient, kvClient);
        if (s.Failure()) [[unlikely]] { return s; }
        return dumpQueue_.Setup(config, &failureSet_, std::move(heteroClient), std::move(kvClient));
    }

protected:
    void Dispatch(TaskPtr task, WaiterPtr waiter) override
    {
        if (task->type == TransTask::Type::LOAD) {
            loadQueue_.Submit(std::move(task), std::move(waiter));
        } else {
            dumpQueue_.Submit(std::move(task), std::move(waiter));
        }
    }
};

}  // namespace UC::YuanRongStore

#endif
