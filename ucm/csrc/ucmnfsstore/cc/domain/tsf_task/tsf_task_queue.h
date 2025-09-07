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
#ifndef UNIFIEDCACHE_TSF_TASK_QUEUE_H
#define UNIFIEDCACHE_TSF_TASK_QUEUE_H

#include "device/idevice.h"
#include "space/space_layout.h"
#include "thread/thread_pool.h"
#include "tsf_task.h"
#include "tsf_task_set.h"

namespace UC {

class TsfTaskQueue {
    int32_t _deviceId{-1};
    size_t _bufferSize{0};
    size_t _bufferNumber{0};
    TsfTaskSet* _failureSet{nullptr};
    const SpaceLayout* _layout{nullptr};
    std::unique_ptr<IDevice> _device{nullptr};
    ThreadPool<TsfTask> _front;
    ThreadPool<std::function<void(void)>> _back;

public:
    Status Setup(const int32_t deviceId, const size_t streamNumber, const size_t bufferSize, const size_t bufferNumber,
                 TsfTaskSet* failureSet, const SpaceLayout* layout);
    void Push(TsfTask&& task);

private:
    bool CreateDevice();
    void Dispatch(TsfTask& task);
    void S2H(TsfTask& task);
    void H2S(TsfTask& task);
    void S2D(TsfTask& task);
    void D2S(TsfTask& task);
    Status Read(const std::string& blockId, const size_t offset, const size_t length, uintptr_t address);
    Status Write(const std::string& blockId, const size_t offset, const size_t length, const uintptr_t address);
    Status AcquireBuffer(uintptr_t* buffer, const size_t number);
    void ReleaseBuffer(uintptr_t* buffer, const size_t number);
};

} // namespace UC

#endif
