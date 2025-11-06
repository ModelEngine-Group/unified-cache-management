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

#include "device/idevice.h"
#include "space/space_layout.h"
#include "status/status.h"
#include "thread/thread_pool.h"
#include "trans_task.h"

namespace UC {

class TransManager {
public:
    Status Setup(const int32_t deviceId, const size_t streamNumber, const size_t ioSize,
                 const size_t bufferNumber, const SpaceLayout* layout, const size_t timeoutMs);
    Status Submit(TransTask task, size_t& taskId) noexcept;
    Status Wait(const size_t taskId) noexcept;
    Status Check(const size_t taskId, bool& finish) noexcept;

private:
    struct DeviceTask {};
    struct FileTask {};
    void DeviceWorker(DeviceTask&);
    void FileWorker(FileTask&);

private:
    std::unique_ptr<IDevice> device_;
    ThreadPool<DeviceTask> devPool_;
    ThreadPool<FileTask> filePool_;
    const SpaceLayout* layout_;
    size_t timeoutMs_;
};

} // namespace UC

#endif
