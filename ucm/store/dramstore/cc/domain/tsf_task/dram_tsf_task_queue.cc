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

#include "dram_tsf_task_queue.h"

namespace UC {

#define UC_TASK_ERROR(s, t)                                                                        \
    do {                                                                                           \
        UC_ERROR("Failed({}) to run task({},{},{},{}).", (s), (t).owner, (t).blockId, (t).offset,  \
                 (t).length);                                                                      \
    } while (0)

Status DramTsfTaskQueue::Setup(const int32_t deviceId, DramTsfTaskSet* failureSet, const MemoryPool* memPool);
{
    this->_failureSet = failureSet;
    this->_memPool = memPool;
    if (deviceId >= 0) {
        this->_device = DeviceFactory::Make(deviceId, 0, 0); // 这里不需要buffer，暂时都先传0吧
        if (!this->_device) { return Status::OutOfMemory(); }
    }
    if (!this->_streamOper.Setup([this](DramTsfTask& task) { this->StreamOper(task); })) {
        return Status::Error();
    }
    return Status::OK();
}

void DramTsfTaskQueue::Push(std::list<DramTsfTask>& tasks)
{
    this->_streamOper.Push(tasks);
}

void DramTsfTaskQueue::StreamOper(DramTsfTask& task)
{
    if (this->_failureSet->Contains(task.owner)) {
        this->Done(task, false);
        return;
    }
    if (task.type == DramTsfTask::Type::LOAD) {
        this->H2D(task);
    } else {
        this->D2H(task);
    }
}

// 这个H2D和D2H函数是重点要重新实现的。
void DramTsfTaskQueue::H2D(DramTsfTask& task)
{
    // TODO 这里地址要重新写逻辑
    auto block_addr = this->memPool_->GetAddress(task.blockId);
    auto host_src = block_addr + task.offset;
    if (!host_src) {
        UC_TASK_ERROR(Status::Error(), task);
        this->Done(task, false);
        return;
    }
    auto status = this->_device->H2DAsync((std::byte*)task.address, host_src, task.length);
    if (status.Failure()) {
        UC_TASK_ERROR(status, task);
        this->Done(task, false);
        return;
    }
    status = this->_device->AppendCallback([this, task](bool success) mutable {
        if (!success) { UC_TASK_ERROR(Status::Error(), task); }
        this->Done(task, success);
        // 这里是否需要return？
    });
    if (status.Failure()) {
        UC_TASK_ERROR(status, task);
        this->Done(task, false);
        return;
    }
}

// 这个函数也是重点要重新实现的。
void DramTsfTaskQueue::D2H(DramTsfTask& task)
{
    // TODO 这里地址要重新写逻辑
    auto block_addr = this->memPool_->GetAddress(task.blockId);
    if (!block_addr) {
        // 如果还没有，那么临时分配
        block_addr = this->memPool_->NewBlock();
        if (!block_addr) {
            UC_TASK_ERROR(Status::Error(), task);
            this->Done(task, false);
            return;
        }
    }
    auto host_dst = block_addr + task.offset;
    if (!host_dst) {
        UC_TASK_ERROR(Status::Error(), task);
        this->Done(task, false);
        return;
    }
    auto status = this->_device->D2HAsync(host_dst, (std::byte*)task.address, task.length);
    if (status.Failure()) {
        UC_TASK_ERROR(status, task);
        this->Done(task, false);
        return;
    }
    status = this->_device->AppendCallback([this, task](bool success) mutable {
        if (!success) {
            UC_TASK_ERROR(Status::Error(), task);
            this->Done(task, false);
            return; // 这里是否需要return？
        }
        this->done(task, true);
    });
    if (status.Failure()) {
        UC_TASK_ERROR(status, task);
        this->Done(task, false);
        return;
    }
}

void DramTsfTaskQueue::Done(const DramTsfTask& task, bool success)
{
    if (!success) { this->_failureSet->Insert(task.owner); }
    task.waiter->Done();
}

} // namespace UC
