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
#include <fcntl.h>
#include <filesystem>
#include <functional>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include "logger/logger.h"
#include "trans/device.h"

namespace UC::MooncakeStore {

static constexpr int32_t SHARE_BUFFER_MAGIC = (('M' << 16) | ('s' << 8) | 1);
static constexpr size_t INVALID_POSITION = size_t(-1);

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
        if (block.size() >= 16) {
            auto data = static_cast<const uint64_t*>((const void*)block.data());
            lo = data[0];
            hi = data[1];
        } else {
            lo = std::hash<std::string>{}(block);
            hi = 0;
        }
    }
    void Reset() { lo = hi = 0; }
    bool Used() const { return lo != 0 || hi != 0; }
    bool operator==(const std::string& block) const
    {
        if (block.size() >= 16) {
            auto data = static_cast<const uint64_t*>((const void*)block.data());
            return lo == data[0] && hi == data[1];
        }
        return lo == std::hash<std::string>{}(block) && hi == 0;
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
    void Refer()
    {
        if (this->ref == 0 && this->status != ShareBlockStatus::LOADED) {
            this->status = ShareBlockStatus::INIT;
        }
        this->ref++;
    }
    void Occupy(const std::string& block)
    {
        this->id.Set(block);
        this->ref = 1;
        this->status = ShareBlockStatus::INIT;
    }
};

struct ShareBufferHeader {
    ShareMutex mutex;
    std::atomic<int32_t> magic;
    size_t blockSize;
    size_t blockNumber;
    ShareBlockHeader headers[0];
};

static const inline std::string& ShmPrefix() noexcept
{
    static std::string prefix{"uc_shm_mooncake_"};
    return prefix;
}

static void CleanUpShmFileExceptMe(const std::string& me)
{
    namespace fs = std::filesystem;
    std::string_view prefix = ShmPrefix();
    fs::path shmDir = "/dev/shm";
    if (!fs::exists(shmDir)) { return; }
    const auto now = fs::file_time_type::clock::now();
    const auto keepThreshold = std::chrono::minutes(10);
    for (const auto& entry : fs::directory_iterator(shmDir)) {
        const auto& path = entry.path();
        const auto& name = path.filename().string();
        if (!entry.is_regular_file() || name.compare(0, prefix.size(), prefix) != 0 || name == me) {
            continue;
        }
        try {
            const auto lwt = fs::last_write_time(path);
            if (now - lwt <= keepThreshold) { continue; }
            fs::remove(path);
        } catch (...) {
        }
    }
}

size_t ShareBuffer::DataOffset() const
{
    static const auto pageSize = sysconf(_SC_PAGESIZE);
    auto headerSize = sizeof(ShareBufferHeader) + sizeof(ShareBlockHeader) * blockNumber_;
    return (headerSize + pageSize - 1) & ~(pageSize - 1);
}

size_t ShareBuffer::ShmSize() const { return DataOffset() + blockSize_ * blockNumber_; }

Status ShareBuffer::Setup(size_t blockSize, size_t blockNumber, const std::string& uniqueId)
{
    blockSize_ = blockSize;
    blockNumber_ = blockNumber;
    addr_ = nullptr;
    tmpBufMaker_ = Trans::Device{}.MakeBuffer();
    if (!tmpBufMaker_) { return Status::OutOfMemory(); }
    shmName_ = ShmPrefix() + uniqueId;
    CleanUpShmFileExceptMe(shmName_);

    static constexpr auto NewFilePerm = (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    int fd = shm_open(shmName_.c_str(), O_RDWR | O_CREAT | O_EXCL, NewFilePerm);
    if (fd >= 0) {
        auto s = InitShmBuffer(fd);
        close(fd);
        return s;
    }
    if (errno == EEXIST) {
        fd = shm_open(shmName_.c_str(), O_RDWR, NewFilePerm);
        if (fd < 0) { return Status{errno, "shm_open failed"}; }
        auto s = LoadShmBuffer(fd);
        close(fd);
        return s;
    }
    return Status{errno, "shm_open failed"};
}

ShareBuffer::~ShareBuffer()
{
    if (!addr_) { return; }
    void* dataAddr = static_cast<char*>(addr_) + DataOffset();
    Trans::Buffer::UnregisterHostBuffer(dataAddr);
    munmap(addr_, ShmSize());
    shm_unlink(shmName_.c_str());
}

Status ShareBuffer::InitShmBuffer(int fd)
{
    const auto shmSize = ShmSize();
    if (ftruncate64(fd, shmSize) != 0) { return Status{errno, "ftruncate failed"}; }
    addr_ = mmap(nullptr, shmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr_ == MAP_FAILED) {
        addr_ = nullptr;
        return Status{errno, "mmap failed"};
    }
    auto bufferHeader = (ShareBufferHeader*)addr_;
    bufferHeader->magic = 1;
    bufferHeader->mutex.Init();
    bufferHeader->blockSize = blockSize_;
    bufferHeader->blockNumber = blockNumber_;
    const auto dataOffset = DataOffset();
    for (size_t i = 0; i < blockNumber_; i++) {
        bufferHeader->headers[i].id.Reset();
        bufferHeader->headers[i].mutex.Init();
        bufferHeader->headers[i].ref = 0;
        bufferHeader->headers[i].status = ShareBlockStatus::INIT;
        const auto headerOffset = sizeof(ShareBufferHeader) + sizeof(ShareBlockHeader) * i;
        bufferHeader->headers[i].offset = dataOffset + blockSize_ * i - headerOffset;
    }
    bufferHeader->magic = SHARE_BUFFER_MAGIC;
    void* dataAddr = static_cast<char*>(addr_) + dataOffset;
    auto dataSize = shmSize - dataOffset;
    auto status = Trans::Buffer::RegisterHostBuffer(dataAddr, dataSize);
    if (status.Success()) { return Status::OK(); }
    UC_ERROR("Failed({}) to register host buffer({}).", status.ToString(), dataSize);
    return Status::Error();
}

Status ShareBuffer::LoadShmBuffer(int fd)
{
    const auto shmSize = ShmSize();
    if (ftruncate64(fd, shmSize) != 0) { return Status{errno, "ftruncate failed"}; }
    addr_ = mmap(nullptr, shmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr_ == MAP_FAILED) {
        addr_ = nullptr;
        return Status{errno, "mmap failed"};
    }
    auto bufferHeader = (ShareBufferHeader*)addr_;
    constexpr auto retryInterval = std::chrono::milliseconds(100);
    constexpr auto maxTryTime = 100;
    auto tryTime = 0;
    do {
        if (bufferHeader->magic == SHARE_BUFFER_MAGIC) { break; }
        if (tryTime > maxTryTime) {
            UC_ERROR("Shm file({}) not ready.", shmName_);
            return Status::Retry();
        }
        std::this_thread::sleep_for(retryInterval);
        tryTime++;
    } while (true);
    const auto dataOffset = DataOffset();
    void* dataAddr = static_cast<char*>(addr_) + dataOffset;
    auto dataSize = shmSize - dataOffset;
    auto status = Trans::Buffer::RegisterHostBuffer(dataAddr, dataSize);
    if (status.Success()) { return Status::OK(); }
    UC_ERROR("Failed({}) to register host buffer({}).", status.ToString(), dataSize);
    return Status::Error();
}

size_t ShareBuffer::AcquireBlock(const std::string& key)
{
    static std::hash<std::string> hasher{};
    auto pos = hasher(key) % blockNumber_;
    auto bufferHeader = (ShareBufferHeader*)addr_;
    auto reusedPos = INVALID_POSITION;
    bufferHeader->mutex.Lock();
    for (size_t i = 0; i < blockNumber_; i++) {
        auto header = bufferHeader->headers + pos;
        header->mutex.Lock();
        if (header->id == key) {
            header->Refer();
            header->mutex.Unlock();
            bufferHeader->mutex.Unlock();
            return pos;
        }
        if (!header->id.Used()) {
            if (reusedPos != INVALID_POSITION) {
                header->mutex.Unlock();
                break;
            }
            header->Occupy(key);
            header->mutex.Unlock();
            bufferHeader->mutex.Unlock();
            return pos;
        }
        if (header->ref <= 0 && reusedPos == INVALID_POSITION) { reusedPos = pos; }
        header->mutex.Unlock();
        pos = (pos + 1) % blockNumber_;
    }
    if (reusedPos != INVALID_POSITION) {
        auto header = bufferHeader->headers + reusedPos;
        header->mutex.Lock();
        header->Occupy(key);
        header->mutex.Unlock();
    }
    bufferHeader->mutex.Unlock();
    return reusedPos;
}

void ShareBuffer::ReleaseBlock(size_t index)
{
    auto bufferHeader = (ShareBufferHeader*)addr_;
    bufferHeader->headers[index].mutex.Lock();
    bufferHeader->headers[index].ref--;
    bufferHeader->headers[index].mutex.Unlock();
}

void* ShareBuffer::BlockAt(size_t index)
{
    auto bufferHeader = (ShareBufferHeader*)addr_;
    return bufferHeader->headers + index;
}

std::shared_ptr<ShareBuffer::Reader> ShareBuffer::MakeReader(const std::string& key)
{
    auto pos = AcquireBlock(key);
    if (pos != INVALID_POSITION) { return MakeSharedReader(key, pos); }
    return MakeLocalReader(key);
}

std::shared_ptr<ShareBuffer::Reader> ShareBuffer::MakeLocalReader(const std::string& key)
{
    auto addr = tmpBufMaker_->MakeHostBuffer(blockSize_);
    if (!addr) [[unlikely]] {
        UC_ERROR("Failed to make buffer({}) on host.", blockSize_);
        return nullptr;
    }
    try {
        auto reader = new Reader{key, blockSize_, false, addr.get()};
        return std::shared_ptr<Reader>(reader, [addr](Reader* r) { delete r; });
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to create reader.", e.what());
        return nullptr;
    }
}

std::shared_ptr<ShareBuffer::Reader> ShareBuffer::MakeSharedReader(const std::string& key,
                                                                   size_t position)
{
    void* addr = BlockAt(position);
    auto reader = new (std::nothrow) Reader(key, blockSize_, true, addr);
    if (!reader) [[unlikely]] {
        ReleaseBlock(position);
        UC_ERROR("Failed to create reader.");
        return nullptr;
    }
    try {
        return std::shared_ptr<Reader>(reader, [this, position](Reader* r) {
            delete r;
            this->ReleaseBlock(position);
        });
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to create reader.", e.what());
        return nullptr;
    }
}

Status ShareBuffer::Reader::Ready4Read()
{
    if (!shared_) { return Status::DuplicateKey(); }
    return Ready4ReadOnSharedBuffer();
}

void ShareBuffer::Reader::MarkLoaded()
{
    if (!shared_) { return; }
    auto header = (ShareBlockHeader*)addr_;
    header->status = ShareBlockStatus::LOADED;
}

void ShareBuffer::Reader::MarkFailed()
{
    if (!shared_) { return; }
    auto header = (ShareBlockHeader*)addr_;
    header->status = ShareBlockStatus::FAILURE;
}

void* ShareBuffer::Reader::GetData()
{
    if (shared_) {
        auto header = (ShareBlockHeader*)addr_;
        return header->Data();
    }
    return addr_;
}

Status ShareBuffer::Reader::Ready4ReadOnSharedBuffer()
{
    auto header = (ShareBlockHeader*)addr_;
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
    // DuplicateKey signals: this process is the owner, must load data into GetData()
    // then call MarkLoaded() or MarkFailed().
    return Status::DuplicateKey();
}

}  // namespace UC::MooncakeStore
