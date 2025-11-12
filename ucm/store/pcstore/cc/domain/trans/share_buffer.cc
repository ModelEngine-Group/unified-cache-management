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
#include "share_buffer.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <unistd.h>
#include "device.h"
#include "file/file.h"
#include "logger/logger.h"

namespace UC {

static constexpr int32_t SHARE_BUFFER_MAGIC = (('S' << 16) | ('b' << 8) | 1);

struct ShareMutex {
    pthread_mutex_t mutex;
    ~ShareMutex() = delete;
    void Init()
    {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ADAPTIVE_NP);
        pthread_mutex_init(&mutex, &attr);
        pthread_mutexattr_destroy(&attr);
    }
    void Lock() { pthread_mutex_lock(&mutex); }
    void Unlock() { pthread_mutex_unlock(&mutex); }
};

struct ShareLock {
    pthread_spinlock_t lock;
    ~ShareLock() = delete;
    void Init() { pthread_spin_init(&lock, PTHREAD_PROCESS_SHARED); }
    void Lock() { pthread_spin_lock(&lock); }
    void Unlock() { pthread_spin_unlock(&lock); }
};

struct ShareBlockId {
    uint64_t lo{0};
    uint64_t hi{0};
    void Set(const std::string& block)
    {
        auto data = static_cast<const uint64_t*>((const void*)block.data());
        lo = data[0];
        hi = data[1];
    }
    void Reset() { lo = hi = 0; }
    bool Used() const { return lo != 0 || hi != 0; }
    bool operator==(const std::string& block) const
    {
        auto data = static_cast<const uint64_t*>((const void*)block.data());
        return lo == data[0] && hi == data[1];
    }
};

enum class ShareBlockStatus { INIT, LOADING, LOADED, FAILURE };

struct ShareBlockHeader {
    ShareBlockId id;
    ShareLock mutex;
    int32_t ref;
    ShareBlockStatus status;
    size_t offset;
    void* Data() { return reinterpret_cast<char*>(this) + offset; }
};

struct ShareBufferHeader {
    ShareMutex mutex;
    std::atomic<int32_t> magic;
    int32_t ref;
    size_t blockSize;
    size_t blockNumber;
    ShareBlockHeader headers[0];
};

inline std::string GenShareBufferName(const size_t blockSize, const size_t blockNumber,
                                      const bool ioDirect, const size_t nSharer)
{
    return fmt::format("uc.buf-{}-{}-{}-{:04x}", blockSize, blockNumber, ioDirect, nSharer);
}

Status ShareBuffer::Setup(const size_t blockSize, const size_t blockNumber, const bool ioDirect,
                          const size_t nSharer)
{
    this->blockSize_ = blockSize;
    this->blockNumber_ = blockNumber;
    this->ioDirect_ = ioDirect;
    this->nSharer_ = nSharer;
    this->addr_ = nullptr;
    this->shmName_ = GenShareBufferName(blockSize, blockNumber, ioDirect, nSharer);
    auto file = File::Make(this->shmName_);
    if (!file) { return Status::OutOfMemory(); }
    auto flags = IFile::OpenFlag::CREATE | IFile::OpenFlag::EXCL | IFile::OpenFlag::READ_WRITE;
    auto s = file->ShmOpen(flags);
    if (s.Success()) { return this->InitShmBuffer(file.get()); }
    if (s == Status::DuplicateKey()) { return this->LoadShmBuffer(file.get()); }
    return s;
}

ShareBuffer::~ShareBuffer()
{
    if (!this->addr_) { return; }
    auto bufferHeader = (ShareBufferHeader*)this->addr_;
    bufferHeader->mutex.Lock();
    auto ref = (--bufferHeader->ref);
    bufferHeader->mutex.Unlock();
    void* dataAddr = static_cast<char*>(this->addr_) + this->DataOffset();
    Device::UnregisterHost(dataAddr);
    const auto shmSize = this->ShmSize();
    File::MUnmap(this->addr_, shmSize);
    if (ref == 0) { File::ShmUnlink(this->shmName_); }
}

std::shared_ptr<ShareBuffer::Reader> ShareBuffer::MakeReader(const std::string& block,
                                                             const std::string& path)
{
    auto index = this->AcquireBlock(block);
    try {
        void* addr = this->BlockAt(index);
        return std::shared_ptr<Reader>(
            new Reader{block, path, blockSize_, ioDirect_, nSharer_, addr},
            [this, index](auto) { this->ReleaseBlock(index); });
    } catch (...) {
        this->ReleaseBlock(index);
        UC_ERROR("Failed to create reader.");
        return nullptr;
    }
}

size_t ShareBuffer::DataOffset() const
{
    static const auto pageSize = sysconf(_SC_PAGESIZE);
    auto headerSize = sizeof(ShareBufferHeader) + sizeof(ShareBlockHeader) * this->blockNumber_;
    return (headerSize + pageSize - 1) & ~(pageSize - 1);
}

size_t ShareBuffer::ShmSize() const
{
    return this->DataOffset() + this->blockSize_ * this->blockNumber_;
}

Status ShareBuffer::InitShmBuffer(IFile* file)
{
    const auto shmSize = this->ShmSize();
    auto s = file->Truncate(shmSize);
    if (s.Failure()) { return s; }
    s = file->MMap(this->addr_, shmSize, true, true, true);
    if (s.Failure()) { return s; }
    auto bufferHeader = (ShareBufferHeader*)this->addr_;
    bufferHeader->magic = 1;
    bufferHeader->mutex.Init();
    bufferHeader->ref = this->nSharer_;
    bufferHeader->blockSize = this->blockSize_;
    bufferHeader->blockNumber = this->blockNumber_;
    const auto dataOffset = this->DataOffset();
    for (size_t i = 0; i < this->blockNumber_; i++) {
        bufferHeader->headers[i].id.Reset();
        bufferHeader->headers[i].mutex.Init();
        bufferHeader->headers[i].ref = 0;
        bufferHeader->headers[i].status = ShareBlockStatus::INIT;
        const auto headerOffset = sizeof(ShareBufferHeader) + sizeof(ShareBlockHeader) * i;
        bufferHeader->headers[i].offset = dataOffset + this->blockSize_ * i - headerOffset;
    }
    bufferHeader->magic = SHARE_BUFFER_MAGIC;
    void* dataAddr = static_cast<char*>(this->addr_) + dataOffset;
    auto dataSize = shmSize - dataOffset;
    return Device::RegisterHost(dataAddr, dataSize);
}

Status ShareBuffer::LoadShmBuffer(IFile* file)
{
    auto s = file->ShmOpen(IFile::OpenFlag::READ_WRITE);
    if (s.Failure()) { return s; }
    const auto shmSize = this->ShmSize();
    s = file->Truncate(shmSize);
    if (s.Failure()) { return s; }
    s = file->MMap(this->addr_, shmSize, true, true, true);
    if (s.Failure()) { return s; }
    auto bufferHeader = (ShareBufferHeader*)this->addr_;
    constexpr auto retryInterval = std::chrono::milliseconds(100);
    constexpr auto maxTryTime = 100;
    auto tryTime = 0;
    do {
        if (bufferHeader->magic == SHARE_BUFFER_MAGIC) { break; }
        if (tryTime > maxTryTime) {
            UC_ERROR("Shm file({}) not ready.", file->Path());
            return Status::Retry();
        }
        std::this_thread::sleep_for(retryInterval);
        tryTime++;
    } while (true);
    const auto dataOffset = this->DataOffset();
    void* dataAddr = static_cast<char*>(this->addr_) + dataOffset;
    auto dataSize = shmSize - dataOffset;
    return Device::RegisterHost(dataAddr, dataSize);
}

size_t ShareBuffer::AcquireBlock(const std::string& block)
{
    static std::hash<std::string> hasher{};
    auto pos = hasher(block) % this->blockNumber_;
    auto bufferHeader = (ShareBufferHeader*)this->addr_;
    auto reusedIdx = this->blockNumber_;
    bufferHeader->mutex.Lock();
    for (size_t i = 0;; i++) {
        if (!bufferHeader->headers[pos].id.Used()) {
            if (reusedIdx == this->blockNumber_) { reusedIdx = pos; }
            break;
        }
        if (bufferHeader->headers[pos].id == block) {
            reusedIdx = pos;
            break;
        }
        if (bufferHeader->headers[pos].ref <= 0) {
            if (reusedIdx == this->blockNumber_) { reusedIdx = pos; }
        }
        pos = (pos + 1) % this->blockNumber_;
        if (i == this->blockNumber_) {
            UC_WARN("Buffer({}) used out.", this->blockNumber_);
            i = 0;
        }
    }
    bufferHeader->headers[reusedIdx].id.Set(block);
    bufferHeader->headers[reusedIdx].ref = this->nSharer_;
    bufferHeader->headers[reusedIdx].status = ShareBlockStatus::INIT;
    bufferHeader->mutex.Unlock();
    return reusedIdx;
}

void ShareBuffer::ReleaseBlock(const size_t index)
{
    auto bufferHeader = (ShareBufferHeader*)this->addr_;
    bufferHeader->headers[index].mutex.Lock();
    bufferHeader->headers[index].ref--;
    bufferHeader->headers[index].mutex.Unlock();
}

void* ShareBuffer::BlockAt(const size_t index)
{
    auto bufferHeader = (ShareBufferHeader*)this->addr_;
    return bufferHeader->headers + index;
}

Status ShareBuffer::Reader::Ready4Read()
{
    auto header = (ShareBlockHeader*)this->addr_;
    if (header->status == ShareBlockStatus::LOADED) { return Status::OK(); }
    if (header->status == ShareBlockStatus::FAILURE) { return Status::Error(); }
    if (header->status == ShareBlockStatus::LOADING) { return Status::Retry(); }
    auto loading = false;
    header->mutex.Lock();
    if (header->status == ShareBlockStatus::INIT) {
        header->status = ShareBlockStatus::LOADING;
        loading = true;
    }
    header->mutex.Unlock();
    if (!loading) { return Status::Retry(); }
    auto s = File::Read(this->path_, 0, this->length_, this->GetData(), this->ioDirect_);
    if (s.Success()) {
        header->status = ShareBlockStatus::LOADED;
        return Status::OK();
    }
    header->status = ShareBlockStatus::FAILURE;
    return s;
}

uintptr_t ShareBuffer::Reader::GetData()
{
    auto header = (ShareBlockHeader*)this->addr_;
    return (uintptr_t)header->Data();
}

} // namespace UC
