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
#include "cache_meta.h"
#include <atomic>
#include <chrono>
#include <thread>
#include "cache_layout.h"
#include "file/file.h"
#include "logger/logger.h"

namespace UC {

static constexpr uint32_t Magic = (('C' << 16) | ('m' << 8) | 1);

struct Header {
    std::atomic<uint32_t> magic;
    uint32_t padding;
    uint64_t reserved[7];
};

CacheMeta::~CacheMeta()
{
    if (this->addr_) {
        File::MUnmap(this->addr_, this->size_);
        File::ShmUnlink(CacheLayout::MetaShmFile());
    }
    this->addr_ = nullptr;
}

Status CacheMeta::Setup(const Index capacity) noexcept
{
    this->index_.Setup(capacity);
    this->hash_.Setup(capacity);
    this->size_ = this->index_.MemorySize() + this->hash_.MemorySize();
    auto file = File::Make(CacheLayout::MetaShmFile());
    if (!file) { return Status::OutOfMemory(); }
    auto openFlags = IFile::OpenFlag::CREATE | IFile::OpenFlag::EXCL | IFile::OpenFlag::READ_WRITE;
    auto status = file->ShmOpen(openFlags);
    if (status.Success()) { return this->InitShmMeta(file.get()); }
    if (status == Status::DuplicateKey()) { return this->LoadShmMeta(file.get()); }
    return status;
}

CacheMeta::Index CacheMeta::Alloc(const std::string& id, const size_t offset) noexcept
{
    auto index = this->index_.Acquire();
    if (index != CacheIndex::npos) {
        this->hash_.Insert(id, offset, index);
        return index;
    }
    auto evict = this->hash_.Evict();
    if (evict != CacheHash::npos) { this->index_.Release(evict); }
    return npos;
}

CacheMeta::Index CacheMeta::Find(const std::string& id, const size_t offset) noexcept
{
    auto idx = this->hash_.Find(id, offset);
    return idx != CacheHash::npos ? idx : npos;
}

void CacheMeta::PutRef(const uint32_t index) noexcept { this->hash_.PutRef(index); }

void CacheMeta::PutRef(const std::string& id, const size_t offset) noexcept
{
    this->hash_.PutRef(id, offset);
}

Status CacheMeta::InitShmMeta(IFile* shmMetaFile)
{
    auto status = shmMetaFile->Truncate(this->size_);
    if (status.Failure()) { return status; }
    status = shmMetaFile->MMap(this->addr_, this->size_, true, true, true);
    if (status.Failure()) { return status; }
    auto header = (Header*)this->addr_;
    auto indexBase = (void*)(((uint8_t*)this->addr_) + sizeof(Header));
    auto hashBase = (void*)(((uint8_t*)indexBase) + this->index_.MemorySize());
    this->index_.Setup(indexBase);
    this->hash_.Setup(hashBase);
    header->padding = 0;
    auto reservedSize = sizeof(header->reserved) / sizeof(*header->reserved);
    std::fill_n(header->reserved, reservedSize, 0);
    header->magic = Magic;
    return Status::OK();
}

Status CacheMeta::LoadShmMeta(IFile* shmMetaFile)
{
    auto openFlags = IFile::OpenFlag::READ_WRITE;
    auto status = shmMetaFile->ShmOpen(openFlags);
    if (status.Failure()) { return status; }
    constexpr auto retryInterval = std::chrono::milliseconds(100);
    constexpr auto maxTryTime = 100;
    auto tryTime = 0;
    IFile::FileStat stat;
    do {
        if (tryTime > maxTryTime) {
            UC_ERROR("Shm file({}) not ready.", shmMetaFile->Path());
            return Status::Retry();
        }
        std::this_thread::sleep_for(retryInterval);
        status = shmMetaFile->Stat(stat);
        if (status.Failure()) { return status; }
        tryTime++;
    } while (static_cast<size_t>(stat.st_size) != this->size_);
    status = shmMetaFile->MMap(this->addr_, this->size_, true, true, true);
    if (status.Failure()) { return status; }
    auto header = (Header*)this->addr_;
    tryTime = 0;
    do {
        if (header->magic == Magic) { break; }
        if (tryTime > maxTryTime) {
            UC_ERROR("Shm file({}) not ready.", shmMetaFile->Path());
            return Status::Retry();
        }
        std::this_thread::sleep_for(retryInterval);
        tryTime++;
    } while (true);
    auto indexBase = (void*)(((uint8_t*)this->addr_) + sizeof(Header));
    auto hashBase = (void*)(((uint8_t*)indexBase) + this->index_.MemorySize());
    this->index_.Setup(indexBase);
    this->hash_.Setup(hashBase);
    return Status::OK();
}

} // namespace UC
