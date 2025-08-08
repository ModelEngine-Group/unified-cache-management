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
#ifndef UNIFIEDCACHE_TSF_TAKS_QUEUE_H
#define UNIFIEDCACHE_TSF_TAKS_QUEUE_H

#include "device/idevice.h"
#include "space/space_layout.h"
#include "thread/thread_pool.h"
#include "tsf_task.h"
#include "tsf_task_set.h"

namespace UC {

class TsfTaskQueue {
public:
    Status Setup(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber, TsfTaskSet* failureSet,
                 const SpaceLayout* layout);
    void Push(std::list<TsfTask>& tasks);

private:
    void StreamOper(TsfTask& task);
    void FileOper(TsfTask& task);
    void H2D(TsfTask& task);
    void D2H(TsfTask& task);
    void H2S(TsfTask& task);
    void S2H(TsfTask& task);
    void Done(const TsfTask& task, bool success);

private:
    ThreadPool<TsfTask> _streamOper;
    ThreadPool<TsfTask> _fileOper;
    std::unique_ptr<IDevice> _device;
    TsfTaskSet* _failureSet;
    const SpaceLayout* _layout;
};

} // namespace UC

#endif
