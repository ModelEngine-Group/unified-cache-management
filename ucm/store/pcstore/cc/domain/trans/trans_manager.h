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
#ifndef UNIFIEDCACHE_TRANS_MANAGER_H
#define UNIFIEDCACHE_TRANS_MANAGER_H

#include "trans_queue.h"
#include "trans_share_queue.h"

namespace UC {

class TransManager {
public:
    Status Setup(const size_t rankSize, const int32_t deviceId, const size_t streamNumber,
                 const size_t blockSize, const size_t ioSize, const bool ioDirect,
                 const size_t bufferNumber, const SpaceLayout* layout, const size_t timeoutMs);
    Status Submit(TransTask task, size_t& taskId) noexcept;
    Status Wait(const size_t taskId) noexcept;
    Status Check(const size_t taskId, bool& finish) noexcept;

private:
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<TaskWaiter>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    TransShareQueue shareQueue_;
    TransQueue queue_;
    size_t rankSize_;
    size_t timeoutMs_;
    std::mutex mutex_;
    std::unordered_map<size_t, TaskPair> tasks_;
    TaskSet failureSet_;
};

} // namespace UC

#endif
