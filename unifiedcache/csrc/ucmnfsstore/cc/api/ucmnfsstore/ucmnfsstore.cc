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
#include "ucmnfsstore.h"
#include <spdlog/fmt/ranges.h>
#include "logger/logger.h"
#include "space/space_manager.h"
#include "status/status.h"
#include "template/singleton.h"
#include "tsf_task/tsf_task_manager.h"
#include "gc/gc_timer.h"

namespace UC {

void ShowSetupParam(const SetupParam& param)
{
    UC_INFO("Set UC::StorageBackends to {}.", param.storageBackends);
    UC_INFO("Set UC::BlockSize to {}.", param.kvcacheBlockSize);
    UC_INFO("Set UC::TransferEnable to {}.", param.transferEnable);
    UC_INFO("Set UC::DeviceId to {}.", param.transferDeviceId);
    UC_INFO("Set UC::StreamNumber to {}.", param.transferStreamNumber);
    UC_INFO("Set UC::IOSize to {}.", param.transferIoSize);
    UC_INFO("Set UC::BufferNumber to {}.", param.transferBufferNumber);
    UC_INFO("Set UC::TimeoutMs to {}.", param.transferTimeoutMs);
    UC_INFO("Set UC::GcEnable to {}.", param.gcEnable);
    UC_INFO("Set UC::GcInterval to {}.", param.gcInterval);
    UC_INFO("Set UC::GcCapacity to {}.", param.gcCapacity);
    UC_INFO("Set UC::GcThresholdPercent to {}.", param.gcThresholdPercent);
    UC_INFO("Set UC::GcCleanupPercent to {}.", param.gcCleanupPercent);
}

int32_t Setup(const SetupParam& param)
{
    auto spaceMgr = Singleton<SpaceManager>::Instance();
    auto status = spaceMgr->Setup(param.storageBackends, param.kvcacheBlockSize);
    if (status.Failure()) {
        UC_ERROR("Failed({}) to setup SpaceManager.", status);
        return status.Underlying();
    }
    if (param.transferEnable) {
        auto taskMgr = Singleton<TsfTaskManager>::Instance();
        status = taskMgr->Setup(param.transferDeviceId, param.transferStreamNumber, param.transferIoSize,
                                param.transferBufferNumber, param.transferTimeoutMs, spaceMgr->GetSpaceLayout());
        if (status.Failure()) {
            UC_ERROR("Failed({}) to setup TsfTaskManager.", status);
            return status.Underlying();
        }
    }
    if (param.gcEnable) {
        auto gcTimer = Singleton<GCTimer>::Instance();
        gcTimer->Setup(param.gcInterval, param.gcCapacity, param.gcThresholdPercent, param.gcCleanupPercent);
        status = gcTimer->Start();
        if (status.Failure()) {
            UC_ERROR("Failed({}) to start GC Timer.", status);
            return status.Underlying();
        }
    }
    ShowSetupParam(param);
    return Status::OK().Underlying();
}

int32_t Alloc(const std::string& blockId)
{
    auto s = Singleton<SpaceManager>::Instance()->NewBlock(blockId);
    if (s.Failure()) {
        UC_ERROR("Failed to allocate kv cache block space, block id: {}, error code: {}.", blockId, s.Underlying());
    }
    return s.Underlying();
}

bool Lookup(const std::string& blockId) { return Singleton<SpaceManager>::Instance()->LookupBlock(blockId); }

size_t Submit(std::list<TsfTask>& tasks, const size_t size, const size_t number, const std::string& brief)
{
    size_t taskId = 0;
    auto status = Singleton<TsfTaskManager>::Instance()->Submit(tasks, size, number, brief, taskId);
    if (status.Failure()) {
        UC_ERROR("Failed({}) to submit tasks.", status);
        return 0;
    }
    return taskId;
}

int32_t Wait(const size_t taskId) { return Singleton<TsfTaskManager>::Instance()->Wait(taskId).Underlying(); }

int32_t Check(const size_t taskId, bool& finish)
{
    return Singleton<TsfTaskManager>::Instance()->Check(taskId, finish).Underlying();
}

void Commit(const std::string& blockId, const bool success)
{
    auto s = Singleton<SpaceManager>::Instance()->CommitBlock(blockId, success);
    if (s.Failure()) {
        UC_ERROR("Failed to commit kv cache block space, block id: {}, error code: {}.", blockId, s.Underlying());
    }
}

} // namespace UC
