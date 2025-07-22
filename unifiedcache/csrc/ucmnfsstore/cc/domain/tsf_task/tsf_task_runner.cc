/**
/* MIT License
/*
/* Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
/*
/* Permission is hereby granted, free of charge, to any person obtaining a copy
/* of this software and associated documentation files (the "Software"), to deal
/* in the Software without restriction, including without limitation the rights
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/* copies of the Software, and to permit persons to whom the Software is
/* furnished to do so, subject to the following conditions:
/*
/* The above copyright notice and this permission notice shall be included in all
/* copies or substantial portions of the Software.
/*
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/* SOFTWARE.
 * */

#include <tsf_task/tsf_task_runner.h>
#include "file/file.h"
#include "logger/logger.h"
#include "memory/memory_allocator.h" 
#include "space/space_layout.h"
#include "template/singleton.h"
#include <map>

namespace UC{

Status TsfTaskRunner::Run(const TsfTask& task, IDevice* device)
{
    switch (task.type) {
        case TsfTask::Type::DUMP:
            if (task.location == TsfTask::Location::DEVICE) {
                return this->SSD2Host(task);
            } else {
                return this->Host2SSD(task);
            }
        case TsfTask::Type::LOAD:
            if (task.location == TsfTask::Location::DEVICE) {
                return this->Host2SSD(task);
            } else {
                return this->SSD2Host(task);
            }
        default:
            return Status::Unsupported();
    }
}

Status TsfTaskRunner::Host2SSD(const TsfTask& task)
{
    auto allocator = MemoryAllocator::Make();
    bool align = allocator->Aligned(task.length) && allocator->Aligned(task.address);
    return this->ReadFile(task, align);
}

Status TsfTaskRunner::SSD2Host(const TsfTask& task)
{
    auto allocator = MemoryAllocator::Make();
    bool align = allocator->Aligned(task.length) && allocator->Aligned(task.address);
    return this->WriteFile(task, align);
}

Status TsfTaskRunner::Device2SSD(const TsfTask& task)
{
    return Status::OK();
}

Status TsfTaskRunner::SSD2Device(const TsfTask& task)
{
    return Status::OK();
}

Status TsfTaskRunner::ReadFile(const TsfTask& task, bool align)
{
    auto spaceLayout = Singleton<SpaceLayout>::Instance();
    auto filePath = spaceLayout.DataFilePath(task.blockId, true);
    auto file = File::Make(filePath);
    if (!file) {
        return Status::OutOfMemory();
    }
    auto mode = align ? OpenMode::RD | OpenMode::DIRECT : OpenMode::RD;
    auto status = file->Open(mode);
    if (status.Failure()) {
        return status;
    }
    return file->Read(reinterpret_cast<void*>(task.address), task.length, task.offset);
}

Status TsfTaskRunner::WriteFile(const TsfTask& task, bool align)
{
    auto spaceLayout = Singleton<SpaceLayout>::Instance();
    auto filePath = spaceLayout.DataFilePath(task.blockId, true);
    auto file = File::Make(filePath);
    if (!file) {
        return Status::OutOfMemory();
    }
    auto mode = align ? OpenMode::WR | OpenMode::DIRECT : OpenMode::WR;
    auto status = file->Open(mode);
    if (status.Failure()) {
        return status;
    }
    return file->Write(reinterpret_cast<const void*>(task.address), task.length, task.offset);
}
} // namespace UC