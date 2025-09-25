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
#include "cache_segment.h"
#include "cache_layout.h"
#include "file/file.h"

namespace UC {

CacheSegment::~CacheSegment()
{
    if (this->base_) {
        File::MUnmap(this->base_, TotalSize());
        File::ShmUnlink(CacheLayout::DataShmFile(this->id_));
    }
    this->base_ = nullptr;
}

Status CacheSegment::Setup(const size_t id, const size_t ioSize)
{
    auto path = CacheLayout::DataShmFile(this->id_);
    auto file = File::Make(path);
    if (!file) { return Status::OutOfMemory(); }
    auto status = Status::OK();
    auto openFlags = IFile::OpenFlag::CREATE | IFile::OpenFlag::READ_WRITE;
    if ((status = file->ShmOpen(openFlags)).Failure()) { return status; }
    auto size = TotalSize();
    if ((status = file->Truncate(size)).Failure()) { return status; }
    if ((status = file->MMap(this->base_, size, true, true, true)).Failure()) { return status; }
    return status;
}

void* CacheSegment::At(const size_t idx) const
{
    auto addr = (uint8_t*)this->base_;
    auto offset = this->ioSize_ * idx;
    return addr + offset;
}

} // namespace UC
