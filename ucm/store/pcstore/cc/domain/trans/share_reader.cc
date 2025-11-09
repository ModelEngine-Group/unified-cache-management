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
#include "share_reader.h"
#include <atomic>
#include <fmt/ranges.h>
#include <semaphore.h>
#include <unistd.h>
#include "file/file.h"

namespace UC {

struct FileCacheHeader {
    std::atomic<int32_t> ref;
    std::atomic_bool loaded;
    std::atomic_bool failure;
    size_t offset;
    auto* Data() { return reinterpret_cast<char*>(this) + offset; }
};

static const auto PAGE_SIZE = sysconf(_SC_PAGESIZE);
static const auto DATA_OFFSET = (sizeof(FileCacheHeader) + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
#define CacheHeader() ((FileCacheHeader*)this->addr_)

ShareReader::ShareReader(const std::string& block, const std::string& path, const size_t length,
                         const bool ioDirect, const size_t nSharer)
{
    this->block_ = "uc." + fmt::format("{:02x}", fmt::join(block, ""));
    this->path_ = path;
    this->length_ = length;
    this->ioDirect_ = ioDirect;
    this->nSharer_ = nSharer;
    this->addr_ = nullptr;
}

ShareReader::~ShareReader()
{
    if (!this->addr_) { return; }
    const auto shmSize = this->ShmSize();
    auto ref = CacheHeader()->ref.fetch_sub(1) - 1;
    File::MUnmap(this->addr_, shmSize);
    if (ref == 0) { File::ShmUnlink(this->block_); }
}

Status ShareReader::Ready4Read()
{
    if (this->addr_) {
        if (CacheHeader()->loaded.load()) { return Status::OK(); }
        if (CacheHeader()->failure.load()) { return Status::Error(); }
        return Status::Retry();
    }
    auto file = File::Make(this->block_);
    if (!file) { return Status::OutOfMemory(); }
    auto flags = IFile::OpenFlag::CREATE | IFile::OpenFlag::EXCL | IFile::OpenFlag::READ_WRITE;
    auto s = file->ShmOpen(flags);
    if (s.Success()) { return this->InitShmBlock(file.get()); }
    if (s == Status::DuplicateKey()) { return this->LoadShmBlock(file.get()); }
    return s;
}

uintptr_t ShareReader::GetData() { return (uintptr_t)(CacheHeader()->Data()); }

size_t ShareReader::ShmSize() const { return DATA_OFFSET + this->length_; }

Status ShareReader::InitShmBlock(IFile* file)
{
    const auto shmSize = this->ShmSize();
    auto s = file->Truncate(shmSize);
    if (s.Failure()) { return s; }
    s = file->MMap(this->addr_, shmSize, true, true, true);
    if (s.Failure()) { return s; }
    CacheHeader()->ref = this->nSharer_;
    CacheHeader()->loaded = false;
    CacheHeader()->failure = false;
    CacheHeader()->offset = DATA_OFFSET;
    s = File::Read(this->path_, 0, this->length_, this->GetData(), this->ioDirect_);
    if (s.Success()) {
        CacheHeader()->loaded = true;
    } else {
        CacheHeader()->failure = true;
    }
    return s;
}

Status ShareReader::LoadShmBlock(IFile* file)
{
    const auto flags = IFile::OpenFlag::READ_WRITE;
    auto s = file->ShmOpen(flags);
    if (s.Failure()) { return s; }
    const auto shmSize = this->ShmSize();
    s = file->MMap(this->addr_, shmSize, true, true, true);
    if (s.Failure()) { return s; }
    if (CacheHeader()->loaded.load()) { return Status::OK(); }
    if (CacheHeader()->failure.load()) { return Status::Error(); }
    return Status::Retry();
}

} // namespace UC
