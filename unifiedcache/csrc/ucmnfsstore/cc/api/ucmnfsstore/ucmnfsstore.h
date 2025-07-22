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
#ifndef UNIFIEDCACHE_NFS_STORE_H
#define UNIFIEDCACHE_NFS_STORE_H

#include <list>
#include <string>
#include <vector>
#include "tsf_task/tsf_task.h"

namespace UC {
/**
 * @brief invalid task id
 * */
#define TRANSFER_INVALID_TASK_ID (size_t(0))

struct SetupParam {
    std::vector<std::string> storageBackends;
    size_t kvcacheBlockSize;
    bool transferEnable;
    int32_t transferDeviceId;
    size_t transferStreamNumber;

    SetupParam(const std::vector<std::string>& storageBackends,
               const size_t kvcacheBlockSize,
               const bool transferEnable)
        : storageBackends{storageBackends}, kvcacheBlockSize{kvcacheBlockSize}, transferEnable{transferEnable},
          transferDeviceId{-1}, transferStreamNumber{256}
    {}
};

int32_t Setup(const SetupParam& param);
int32_t Alloc(const std::string& blockId);
bool Lookup(const std::string& blockId);
size_t Submit(std::list<TsfTask> tasks);
int32_t Wait(const size_t taskId);      
void Commit(const std::string& blockId, const bool success);

} // namespace UC

#endif
