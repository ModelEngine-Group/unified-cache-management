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
#ifndef UNIFIEDCACHE_DRAM_TSF_TAKS_QUEUE_H
#define UNIFIEDCACHE_DRAM_TSF_TAKS_QUEUE_H

#include "idevice.h"
#include "thread/thread_pool.h"
#include "dram_tsf_task.h"
#include "dram_tsf_task_set.h"
#include "memory/memory_pool.h"

namespace UC {

class DramTsfTaskQueue {
public:
    Status Setup(const int32_t deviceId, 
                 DramTsfTaskSet* failureSet, MemoryPool* memPool);
    void Push(std::list<DramTsfTask>& tasks);

private:
    void StreamOper(DramTsfTask& task);
    void H2D(DramTsfTask& task);
    void D2H(DramTsfTask& task);
    void Done(const DramTsfTask& task, bool success);

private:
    ThreadPool<DramTsfTask> _streamOper;
    std::unique_ptr<IDevice> _device;
    DramTsfTaskSet* _failureSet;
    MemoryPool* _memPool;
};

} // namespace UC

#endif
