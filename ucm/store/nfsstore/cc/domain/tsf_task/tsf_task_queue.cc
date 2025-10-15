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
#include "tsf_task_queue.h"
#include "file/file.h"

namespace UC {

#define UC_TASK_ERROR(s, t)                                                                        \
    do {                                                                                           \
        UC_ERROR("Failed({}) to run task({},{},{},{}).", (s), (t).owner, (t).blockId, (t).offset,  \
                 (t).length);                                                                      \
    } while (0)

Status TsfTaskQueue::Setup(const int32_t deviceId, const size_t bufferSize,
                           const size_t bufferNumber, TsfTaskSet* failureSet,
                           const SpaceLayout* layout, bool transferUseDirect)
{
    this->_failureSet = failureSet;
    this->_layout = layout;
    if (deviceId >= 0) {
        this->_device = DeviceFactory::Make(deviceId, bufferSize, bufferNumber);
        if (!this->_device) { return Status::OutOfMemory(); }
        if (!this->_streamOper.Setup([this](TsfTask& task) { this->StreamOper(task); },
                                     [this,transferUseDirect] { return this->_device->Setup(transferUseDirect).Success(); })) {
            return Status::Error();
        }
    }
    if (!this->_fileOper.Setup([this](TsfTask& task) { this->FileOper(task); })) {
        return Status::Error();
    }
    return Status::OK();
}

void TsfTaskQueue::Push(std::list<TsfTask>& tasks)
{
    auto& front = tasks.front();
    if (front.location == TsfTask::Location::HOST || front.type == TsfTask::Type::LOAD) {
        this->_fileOper.Push(tasks);
    } else {
        this->_streamOper.Push(tasks);
    }
}

void TsfTaskQueue::StreamOper(TsfTask& task)
{
    if (this->_failureSet->Contains(task.owner)) {
        this->Done(task, false);
        return;
    }
    if (task.type == TsfTask::Type::LOAD) {
        this->H2D(task);
    } else {
        this->D2H(task);
    }
}

void TsfTaskQueue::FileOper(TsfTask& task)
{
    if (this->_failureSet->Contains(task.owner)) {
        this->Done(task, false);
        return;
    }
    if (task.type == TsfTask::Type::LOAD) {
        this->S2H(task);
    } else {
        this->H2S(task);
    }
}

void TsfTaskQueue::H2D(TsfTask& task)
{
    auto status = this->_device->H2DAsync((std::byte*)task.address, task.hub.get(), task.length);
    if (status.Failure()) {
        UC_TASK_ERROR(status, task);
        this->Done(task, false);
        return;
    }
    status = this->_device->AppendCallback([this, task](bool success) mutable {
        if (!success) { UC_TASK_ERROR(Status::Error(), task); }
        this->Done(task, success);
    });
    if (status.Failure()) {
        UC_TASK_ERROR(status, task);
        this->Done(task, false);
        return;
    }
}

void TsfTaskQueue::D2H(TsfTask& task)
{
    task.hub = this->_device->GetBuffer(task.length);
    if (!task.hub) {
        UC_TASK_ERROR(Status::OutOfMemory(), task);
        this->Done(task, false);
        return;
    }
    auto status = this->_device->D2HAsync(task.hub.get(), (std::byte*)task.address, task.length);
    if (status.Failure()) {
        UC_TASK_ERROR(status, task);
        this->Done(task, false);
        return;
    }
    status = this->_device->AppendCallback([this, task](bool success) mutable {
        if (success) {
            this->_fileOper.Push(std::move(task));
        } else {
            UC_TASK_ERROR(Status::Error(), task);
            this->Done(task, false);
            return;
        }
    });
    if (status.Failure()) {
        UC_TASK_ERROR(status, task);
        this->Done(task, false);
        return;
    }
}

void TsfTaskQueue::H2S(TsfTask& task)
{
    const void* src =
        task.location == TsfTask::Location::HOST ? (void*)task.address : task.hub.get();
    auto path = this->_layout->DataFilePath(task.blockId, true);
    auto status = Status::OK();
    do {
        auto file = File::Make(path);
        if (!file) {
            status = Status::OutOfMemory();
            break;
        }
        if ((status = file->Open(IFile::OpenFlag::WRITE_ONLY)).Failure()) { break; }
        if ((status = file->Write(src, task.length, task.offset)).Failure()) { break; }
    } while (0);
    if (status.Failure()) {
        UC_TASK_ERROR(status, task);
        this->Done(task, false);
        return;
    }
    this->Done(task, true);
}

void TsfTaskQueue::S2H(TsfTask& task)
{
    auto path = this->_layout->DataFilePath(task.blockId, false);
    auto status = Status::OK();
    do {
        auto dst = (void*)task.address;
        if (task.location == TsfTask::Location::DEVICE) {
            if (!(task.hub = this->_device->GetBuffer(task.length))) {
                status = Status::OutOfMemory();
                break;
            }
            dst = task.hub.get();
        }
        auto file = File::Make(path);
        if (!file) {
            status = Status::OutOfMemory();
            break;
        }
        if ((status = file->Open(IFile::OpenFlag::READ_ONLY)).Failure()) { break; }
        if ((status = file->Read(dst, task.length, task.offset)).Failure()) { break; }
    } while (0);
    if (status.Failure()) {
        UC_TASK_ERROR(status, task);
        this->Done(task, false);
        return;
    }
    if (task.location == TsfTask::Location::HOST) {
        this->Done(task, true);
        return;
    }
    this->_streamOper.Push(std::move(task));
}

void TsfTaskQueue::Done(const TsfTask& task, bool success)
{
    if (!success) { this->_failureSet->Insert(task.owner); }
    task.waiter->Done();
}

} // namespace UC
