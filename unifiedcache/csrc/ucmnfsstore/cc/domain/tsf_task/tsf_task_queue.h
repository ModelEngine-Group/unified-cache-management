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
#ifndef UNIFIEDCACHE_TSF_TASK_QUEUE
#define UNIFIEDCACHE_TSF_TASK_QUEUE

#include <condition_variable>
#include <future>
#include <list>
#include <mutex>
#include <thread>
#include "tsf_task_runner.h"
#include "tsf_task_set.h"

namespace UC{

class TsfTaskQueue{
public:
    ~TsfTaskQueue();
    Status Setup(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber, TsfTaskSet* failureSet);
    void Push(std::list<TsfTask>& tasks);
    void Push(TsfTask&& task);
    bool Finish(const size_t& taskId) const;

private:
    void Worker();

private:
    int32_t _deviceId{-1};
    size_t _bufferSize{0};
    size_t _bufferNumber{0};
    TsfTaskSet* _failureSet{nullptr};
    bool _running{false};
    std::list<TsfTask> _q;
    std::thread _worker;
    std::promise<Status> _started;
    std::mutex _mutex;
    std::condition_variable _cv;
    TsfTaskRunner _runner;
    size_t _lastId{0};
};

}// namespace UC

#endif