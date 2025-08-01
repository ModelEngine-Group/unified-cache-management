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

#include "tsf_task_runner.h"
#include <functional>
#include <map>
#include "file/file.h"
#include "logger/logger.h"
#include "memory/memory.h"
#include "space/space_manager.h"
#include "template/singleton.h"

namespace UC {

Status TsfTaskRunner::Run(const TsfTask& task)
{
    using Location = TsfTask::Location;
    using Type = TsfTask::Type;
    using Key = std::pair<Location, Type>;
    using Value = std::function<Status(const TsfTask&)>;

    std::map<Key, Value> runners = {
        {{Location::HOST, Type::LOAD}, [this](const TsfTask& task) { return this->Ssd2Host(task); }},
        {{Location::HOST, Type::DUMP}, [this](const TsfTask& task) { return this->Host2Ssd(task); }},
        {{Location::DEVICE, Type::LOAD}, [this](const TsfTask& task) { return this->Ssd2Device(task); }},
        {{Location::DEVICE, Type::DUMP}, [this](const TsfTask& task) { return this->Device2Ssd(task); }},
    };
    auto runner = runners.find({task.location, task.type});
    if (runner == runners.end()) {
        UC_ERROR("Unsupported task({},{})", fmt::underlying(task.location), fmt::underlying(task.type));
        return Status::Unsupported();
    }
    return runner->second(task);
}

Status TsfTaskRunner::Ssd2Host(const TsfTask& task)
{
    auto path = Singleton<SpaceManager>::Instance()->BlockPath(task.blockId);
    auto file = File::Make(path);
    if (!file) { return Status::OutOfMemory(); }
    auto aligned = Memory::Aligned(task.address) && Memory::Aligned(task.offset) && Memory::Aligned(task.length);
    auto openFlags = aligned ? IFile::OpenFlag::READ_ONLY | IFile::OpenFlag::DIRECT : IFile::OpenFlag::READ_ONLY;
    auto status = file->Open(openFlags);
    if (status.Failure()) { return status; }
    return file->Read((void*)task.address, task.length, task.offset);
}

Status TsfTaskRunner::Host2Ssd(const TsfTask& task)
{
    auto path = Singleton<SpaceManager>::Instance()->BlockPath(task.blockId, true);
    auto file = File::Make(path);
    if (!file) { return Status::OutOfMemory(); }
    auto aligned = Memory::Aligned(task.address) && Memory::Aligned(task.offset) && Memory::Aligned(task.length);
    auto openFlags = aligned ? IFile::OpenFlag::WRITE_ONLY | IFile::OpenFlag::DIRECT : IFile::OpenFlag::WRITE_ONLY;
    auto status = file->Open(openFlags);
    if (status.Failure()) { return status; }
    return file->Write((void*)task.address, task.length, task.offset);
}

Status TsfTaskRunner::Ssd2Device(const TsfTask& task)
{
    auto path = Singleton<SpaceManager>::Instance()->BlockPath(task.blockId);
    auto file = File::Make(path);
    if (!file) { return Status::OutOfMemory(); }
    auto buffer = this->_device->GetHostBuffer(task.length);
    if (!buffer) { 
        UC_ERROR("Failed to get host buffer({}) on device", task.length);
        return Status::OutOfMemory(); 
    }
    auto aligned = Memory::Aligned(task.offset) && Memory::Aligned(task.length);
    auto openFlags = aligned ? IFile::OpenFlag::READ_ONLY | IFile::OpenFlag::DIRECT : IFile::OpenFlag::READ_ONLY;
    auto status = file->Open(openFlags);
    if (status.Failure()) { return status; }
    status = file->Read(buffer.get(), task.length, task.offset);
    if (status.Failure()) { return status; }
    return this->_device->H2DAsync((void*)task.address, task.length, buffer.get(), task.length);
}

Status TsfTaskRunner::Device2Ssd(const TsfTask& task)
{
    auto path = Singleton<SpaceManager>::Instance()->BlockPath(task.blockId, true);
    auto file = File::Make(path);
    if (!file) { return Status::OutOfMemory(); }
    auto buffer = this->_device->GetHostBuffer(task.length);
    if (!buffer) {
        UC_ERROR("Failed to get host buffer({}) on device", task.length);
        return Status::OutOfMemory(); 
    }
    auto status = this->_device->D2HAsync(buffer.get(), task.length, (void*)task.address, task.length);
    if (status.Failure()) { return status; }
    status = this->_device->WaitFinish();
    if (status.Failure()) { return status; }
    auto aligned = Memory::Aligned(task.offset) && Memory::Aligned(task.length);
    auto openFlags = aligned ? IFile::OpenFlag::WRITE_ONLY | IFile::OpenFlag::DIRECT : IFile::OpenFlag::WRITE_ONLY;  
    status = file->Open(openFlags);
    if (status.Failure()) { return status; }
    return file->Write(buffer.get(), task.length, task.offset);
}

} // namespace UC