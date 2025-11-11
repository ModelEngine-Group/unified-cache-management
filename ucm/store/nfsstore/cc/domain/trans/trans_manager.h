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
#include "directstorage_queue.h"
#include "infra/template/handle_recorder.h"
#include <unistd.h>

namespace UC {

class TransManager : public TaskManager {
private:
    HandlePool<std::string, int> handlePool_;

public:
    Status Setup(const int32_t deviceId, const size_t streamNumber, const size_t ioSize,
                 const size_t bufferNumber, const SpaceLayout* layout, const size_t timeoutMs, bool useDirect)
    {
        this->timeoutMs_ = timeoutMs;
        auto status = Status::OK();

        status = DeviceFactory::Setup(useDirect);
        if (status.Failure()) {
            UC_ERROR("Failed to setup device factory");
            return status;
        }

        for (size_t i = 0; i < streamNumber; i++) {
            std::shared_ptr<TaskQueue> q;

            if(useDirect) {
                auto directQ = std::make_shared<DirectStorageQueue>();
                status = directQ->Setup(deviceId, ioSize, bufferNumber, &this->failureSet_, layout, timeoutMs, &handlePool_);
                q = directQ;
            }
            else {
                auto posixQ = std::make_shared<PosixQueue>();
                status = posixQ->Setup(deviceId, ioSize, bufferNumber, &this->failureSet_, layout, timeoutMs);
                q = posixQ;
            }

            if (status.Failure()) { break; }
            this->queues_.emplace_back(std::move(q));
        }
        return status;
    }

    ~TransManager() {
        handlePool_.ClearAll([](int fd) {
            if (fd >= 0) {
                close(fd);
            }
        });
    }
};

} // namespace UC

#endif
