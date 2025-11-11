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

#include "posix_queue.h"
#include "task_manager.h"

namespace UC {

class TransManager : public TaskManager {
public:
    Status Setup(const int32_t deviceId, const size_t streamNumber, const size_t ioSize,
                 const size_t bufferNumber, const SpaceLayout* layout, const size_t timeoutMs, bool useDirect = false)
    {
        this->timeoutMs_ = timeoutMs;
        auto status = Status::OK();
        for (size_t i = 0; i < streamNumber; i++) {
            auto q = std::make_shared<PosixQueue>();
            status =
                q->Setup(deviceId, ioSize, bufferNumber, &this->failureSet_, layout, timeoutMs, useDirect);
            if (status.Failure()) { break; }
            this->queues_.emplace_back(std::move(q));
        }
        return status;
    }
};

} // namespace UC

#endif
