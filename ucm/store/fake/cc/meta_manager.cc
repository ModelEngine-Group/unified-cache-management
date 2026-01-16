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
#include "meta_manager.h"
#include <atomic>
#include <filesystem>
#include <thread>
#include "logger/logger.h"
#include "posix_shm.h"

namespace UC::FakeStore {

static constexpr size_t nHashTableBucket = 16411;
static constexpr auto invalidIndex = std::numeric_limits<size_t>::max();

static inline size_t Hash(const Detail::BlockId& blockId)
{
    static UC::Detail::BlockIdHasher blockIdHasher;
    return blockIdHasher(blockId) % nHashTableBucket;
}

struct BufferMetaNode {
    Detail::BlockId block;
    size_t hash;
    size_t prev;
    size_t next;
    void Init()
    {
        hash = invalidIndex;
        prev = invalidIndex;
        next = invalidIndex;
    }
};

class MetaStrategy {
public:
    virtual ~MetaStrategy() = default;
    virtual Status Setup(const std::string& uuid, size_t nNode) = 0;
    virtual void BucketLock(size_t iBucket) = 0;
    virtual bool BucketTryLock(size_t iBucket) = 0;
    virtual void BucketUnlock(size_t iBucket) = 0;
    virtual void NodeLock(size_t iNode) = 0;
    virtual void NodeUnlock(size_t iNode) = 0;
    virtual size_t& FirstAt(size_t iBucket) = 0;
    virtual size_t FetchNode() = 0;
    virtual BufferMetaNode* MetaAt(size_t iNode) = 0;
};

class SharedMetaStrategy : public MetaStrategy {
protected:
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
        bool TryLock() { return pthread_mutex_trylock(&mutex) == 0; }
        void Unlock() { pthread_mutex_unlock(&mutex); }
    };
    struct ShareLock {
        pthread_spinlock_t lock;
        ~ShareLock() = delete;
        void Init() { pthread_spin_init(&lock, PTHREAD_PROCESS_SHARED); }
        void Lock() { pthread_spin_lock(&lock); }
        bool TryLock() { return pthread_spin_trylock(&lock) == 0; }
        void Unlock() { pthread_spin_unlock(&lock); }
    };
    static constexpr size_t sharedBufferMagic = (('S' << 16) | ('b' << 8) | 1);
    struct BufferHeader {
        std::atomic<size_t> magic;
        ShareLock lock;
        size_t freeHead;
        size_t buckets[nHashTableBucket];
        ShareMutex bucketLocks[nHashTableBucket];
        ShareLock nodeLocks[0];
    };

    BufferHeader* header_{nullptr};
    BufferMetaNode* meta_{nullptr};
    std::string shmName_;
    size_t nNode_;
    void* addrress_{nullptr};
    size_t totalSize_;

    size_t MetaOffset() const noexcept { return sizeof(BufferHeader) + sizeof(ShareLock) * nNode_; }
    size_t MetaSize() const noexcept { return sizeof(BufferMetaNode) * nNode_; }
    const std::string& ShmPrefix() const noexcept
    {
        static std::string prefix{"uc_shm_fake_"};
        return prefix;
    }
    void CleanUpShmFileExceptMe()
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
            if (!entry.is_regular_file() || name.compare(0, prefix.size(), prefix) != 0 ||
                name == shmName_) {
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
    Status MmapShmFile(PosixShm& shmFile, bool needTrunc = true)
    {
        auto s = Status::OK();
        if (needTrunc) {
            s = shmFile.Truncate(totalSize_);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to truncate file({}) with size({}).", s, shmName_, totalSize_);
                return s;
            }
        }
        s = shmFile.MMap(addrress_, totalSize_, true, true, true);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to mmap file({}) with size({}).", s, shmName_, totalSize_);
            return s;
        }
        header_ = (BufferHeader*)addrress_;
        return Status::OK();
    }
    Status InitShmBuffer(PosixShm& shmFile)
    {
        auto s = MmapShmFile(shmFile);
        if (s.Failure()) [[unlikely]] { return s; }
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        header_->lock.Init();
        header_->freeHead = 0;
        for (size_t i = 0; i < nHashTableBucket; i++) {
            header_->buckets[i] = invalidIndex;
            header_->bucketLocks[i].Init();
        }
        for (size_t i = 0; i < nNode_; i++) {
            header_->nodeLocks[i].Init();
            meta_[i].Init();
        }
        header_->magic = sharedBufferMagic;
        return Status::OK();
    }
    Status LoadShmBuffer(PosixShm& shmFile)
    {
        auto s = shmFile.ShmOpen(PosixShm::OpenFlag::READ_WRITE);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to open file({}).", s, shmName_);
            return s;
        }
        s = MmapShmFile(shmFile, false);
        if (s.Failure()) [[unlikely]] { return s; }
        constexpr auto retryInterval = std::chrono::milliseconds(100);
        constexpr auto maxTryTime = 100;
        auto tryTime = 0;
        do {
            if (header_->magic == sharedBufferMagic) { break; }
            if (tryTime > maxTryTime) {
                UC_ERROR("Shm file({}) not ready.", shmName_);
                return Status::Retry();
            }
            std::this_thread::sleep_for(retryInterval);
            tryTime++;
        } while (true);
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        return Status::OK();
    }

public:
    ~SharedMetaStrategy() override
    {
        if (addrress_) { PosixShm::MUnmap(addrress_, totalSize_); }
        PosixShm{shmName_}.ShmUnlink();
    }
    Status Setup(const std::string& uuid, size_t nNode) override
    {
        shmName_ = ShmPrefix() + uuid;
        nNode_ = nNode;
        totalSize_ = MetaOffset() + MetaSize();
        CleanUpShmFileExceptMe();
        PosixShm shmFile{shmName_};
        const auto flags =
            PosixShm::OpenFlag::CREATE | PosixShm::OpenFlag::EXCL | PosixShm::OpenFlag::READ_WRITE;
        auto s = shmFile.ShmOpen(flags);
        if (s.Success()) {
            s = InitShmBuffer(shmFile);
        } else if (s == Status::DuplicateKey()) {
            s = LoadShmBuffer(shmFile);
        } else {
            UC_ERROR("Failed({}) to open file({}) with flags({}).", s, shmName_, flags);
        }
        return s;
    }
    void BucketLock(size_t iBucket) override { header_->bucketLocks[iBucket].Lock(); }
    bool BucketTryLock(size_t iBucket) override { return header_->bucketLocks[iBucket].TryLock(); }
    void BucketUnlock(size_t iBucket) override { header_->bucketLocks[iBucket].Unlock(); }
    void NodeLock(size_t iNode) override { header_->nodeLocks[iNode].Lock(); }
    void NodeUnlock(size_t iNode) override { header_->nodeLocks[iNode].Unlock(); }
    size_t& FirstAt(size_t iBucket) override { return header_->buckets[iBucket]; }
    size_t FetchNode() override
    {
        header_->lock.Lock();
        auto iNode = header_->freeHead++;
        if (header_->freeHead == nNode_) { header_->freeHead = 0; }
        header_->lock.Unlock();
        return iNode;
    }
    BufferMetaNode* MetaAt(size_t iNode) override { return meta_ + iNode; }
};

Status MetaManager::Setup(const Config& config)
{
    try {
        strategy_ = std::make_shared<SharedMetaStrategy>();
    } catch (const std::exception& e) {
        return Status::Error(fmt::format("failed({}) to make buffer strategy", e.what()));
    }
    return strategy_->Setup(config.uniqueId, config.bufferNumber);
}

void MetaManager::Insert(const Detail::BlockId& block) noexcept
{
    auto iBucket = Hash(block);
    strategy_->BucketLock(iBucket);
    if (!ExistAt(iBucket, block)) { InsertAt(iBucket, block); }
    strategy_->BucketUnlock(iBucket);
    return;
}

bool MetaManager::Exist(const Detail::BlockId& block) const noexcept
{
    auto iBucket = Hash(block);
    strategy_->BucketLock(iBucket);
    auto exist = ExistAt(iBucket, block);
    strategy_->BucketUnlock(iBucket);
    return exist;
}

bool MetaManager::ExistAt(size_t iBucket, const Detail::BlockId& block) const noexcept
{
    auto iNode = strategy_->FirstAt(iBucket);
    while (iNode != invalidIndex) {
        auto meta = strategy_->MetaAt(iNode);
        strategy_->NodeLock(iNode);
        if (meta->block == block) {
            strategy_->NodeUnlock(iNode);
            return true;
        }
        auto next = meta->next;
        strategy_->NodeUnlock(iNode);
        iNode = next;
    }
    return false;
}

void MetaManager::InsertAt(size_t iBucket, const Detail::BlockId& block) noexcept
{
    for (;;) {
        auto iNode = strategy_->FetchNode();
        strategy_->NodeLock(iNode);
        auto meta = strategy_->MetaAt(iNode);
        const auto oldBucket = meta->hash;
        if (oldBucket != iBucket) {
            if (oldBucket != invalidIndex) {
                if (!strategy_->BucketTryLock(oldBucket)) {
                    strategy_->NodeUnlock(iNode);
                    continue;
                }
                Remove(oldBucket, iNode);
                strategy_->BucketUnlock(oldBucket);
            }
            MoveTo(iBucket, iNode);
        }
        meta->block = block;
        strategy_->NodeUnlock(iNode);
        return;
    }
}

void MetaManager::MoveTo(size_t iBucket, size_t iNode) noexcept
{
    auto meta = strategy_->MetaAt(iNode);
    auto& head = strategy_->FirstAt(iBucket);
    auto n = head;
    meta->next = n;
    if (n != invalidIndex) {
        auto next = strategy_->MetaAt(n);
        strategy_->NodeLock(n);
        next->prev = iNode;
        strategy_->NodeUnlock(n);
    }
    meta->hash = iBucket;
    head = iNode;
}

void MetaManager::Remove(size_t iBucket, size_t iNode) noexcept
{
    auto meta = strategy_->MetaAt(iNode);
    auto p = meta->prev;
    if (p != invalidIndex) {
        auto prev = strategy_->MetaAt(p);
        strategy_->NodeLock(p);
        prev->next = meta->next;
        strategy_->NodeUnlock(p);
    }
    auto n = meta->next;
    if (n != invalidIndex) {
        auto next = strategy_->MetaAt(n);
        strategy_->NodeLock(n);
        next->prev = meta->prev;
        strategy_->NodeUnlock(n);
    }
    if (strategy_->FirstAt(iBucket) == iNode) { strategy_->FirstAt(iBucket) = n; }
    meta->prev = meta->next = invalidIndex;
    meta->hash = invalidIndex;
}

}  // namespace UC::FakeStore
