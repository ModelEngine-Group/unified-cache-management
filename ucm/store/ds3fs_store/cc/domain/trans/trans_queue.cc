#include "trans_queue.h"
#include "file/file.h"
#include "logger/logger.h"
#include "trans/device.h"
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <hf3fs_usrbio.h>
#include <thread>

namespace UC {

thread_local void* g_threadUsrbioResources = nullptr;

TransQueue::~TransQueue()
{
    CleanupUsrbio();
}

void TransQueue::DeviceWorker(BlockTask&& task)
{
    if (this->failureSet_->Contains(task.owner)) {
        task.done(false);
        return;
    }

    auto number = task.shards.size();
    auto size = this->ioSize_;
    auto done = task.done;
    auto devPtrs = (void**)task.shards.data();
    auto hostPtr = task.buffer.get();
    auto s = Status::OK();

    if (task.type == TransTask::Type::LOAD) {
        s = stream_->HostToDevice(hostPtr, devPtrs, size, number);
    } else {
        s = stream_->DeviceToHost(devPtrs, hostPtr, size, number);
        if (s.Success()) {
            this->filePool_.Push(std::move(task));
        }
    }

    if (s.Failure()) {
        this->failureSet_->Insert(task.owner);
    }

    done(s.Success());
}

static inline void* memcpy_fast(void* dst, const void* src, size_t size)
{
    if (size == 0) return dst;

    size_t aligned_size = size & ~(sizeof(uint64_t) - 1);
    if (aligned_size > 0) {
        uint64_t* d = (uint64_t*)dst;
        const uint64_t* s = (const uint64_t*)src;
        size_t count = aligned_size / sizeof(uint64_t);
        for (size_t i = 0; i < count; ++i) {
            d[i] = s[i];
        }
    }

    size_t remaining = size - aligned_size;
    if (remaining > 0) {
        uint8_t* d = (uint8_t*)dst + aligned_size;
        const uint8_t* s = (const uint8_t*)src + aligned_size;
        for (size_t i = 0; i < remaining; ++i) {
            d[i] = s[i];
        }
    }

    return dst;
}

void TransQueue::FileWorker(BlockTask&& task)
{
    if (this->failureSet_->Contains(task.owner)) {
        task.done(false);
        return;
    }

    UsrbioResources& usrbio_ = GetThreadUsrbioResources();

    auto hostPtr = (uintptr_t)task.buffer.get();
    auto length = this->ioSize_ * task.shards.size();

    if (task.type == TransTask::Type::DUMP) {
        const auto& path = this->layout_->DataFilePath(task.block, true);

        FdHandle fdHandle{};
        auto status = fdHandlePool_.Get(path, fdHandle, [&path](FdHandle& handle) -> Status {
            int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) {
                return Status::Error();
            }

            if (!hf3fs_is_hf3fs(fd)) {
                close(fd);
                return Status::Error();
            }

            int regRes = hf3fs_reg_fd(fd, 0);
            if (regRes > 0) {
                close(fd);
                return Status::Error();
            }

            handle.fd = fd;
            handle.regFd = regRes;
            return Status::OK();
        });

        if (status.Failure()) {
            this->failureSet_->Insert(task.owner);
            task.done(false);
            return;
        }

        int regFd = fdHandle.regFd;

        if (hostPtr > 0 && length > 0) {
            if (length > usrbio_.writeIov.size) {
                this->failureSet_->Insert(task.owner);
                task.done(false);
                return;
            }
            memcpy_fast(usrbio_.writeIov.base, (void*)hostPtr, length);
        }

        int prepRes = hf3fs_prep_io(&usrbio_.writeIor,
                                   &usrbio_.writeIov,
                                   false,
                                   usrbio_.writeIov.base,
                                   regFd,
                                   0,
                                   length,
                                   (const void*)(uintptr_t)task.owner);

        if (prepRes < 0) {
            this->failureSet_->Insert(task.owner);
            task.done(false);
            return;
        }

        int submitRes = hf3fs_submit_ios(&usrbio_.writeIor);
        if (submitRes < 0) {
            this->failureSet_->Insert(task.owner);
            task.done(false);
            return;
        }

        struct hf3fs_cqe cqe;
        int waitRes = hf3fs_wait_for_ios(&usrbio_.writeIor,
                                        &cqe,
                                        1,
                                        1,
                                        nullptr);

        if (waitRes <= 0) {
            this->failureSet_->Insert(task.owner);
            task.done(false);
            return;
        }

        if (cqe.result < 0) {
            this->failureSet_->Insert(task.owner);
            task.done(false);
            return;
        }

        this->layout_->Commit(task.block, true);

        task.done(true);
        return;
    }

    const auto& path = this->layout_->DataFilePath(task.block, false);

    FdHandle fdHandle{};
    auto status = fdHandlePool_.Get(path, fdHandle, [&path](FdHandle& handle) -> Status {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            return Status::Error();
        }

        if (!hf3fs_is_hf3fs(fd)) {
            close(fd);
            return Status::Error();
        }

        int regRes = hf3fs_reg_fd(fd, 0);
        if (regRes > 0) {
            close(fd);
            return Status::Error();
        }

        handle.fd = fd;
        handle.regFd = regRes;
        return Status::OK();
    });

    if (status.Failure()) {
        this->failureSet_->Insert(task.owner);
        task.done(false);
        return;
    }

    int regFd = fdHandle.regFd;

    if (length > usrbio_.readIov.size) {
        this->failureSet_->Insert(task.owner);
        task.done(false);
        return;
    }

    int prepRes = hf3fs_prep_io(&usrbio_.readIor,
                               &usrbio_.readIov,
                               true,
                               usrbio_.readIov.base,
                               regFd,
                               0,
                               length,
                               (const void*)(uintptr_t)task.owner);

    if (prepRes < 0) {
        this->failureSet_->Insert(task.owner);
        task.done(false);
        return;
    }

    int submitRes = hf3fs_submit_ios(&usrbio_.readIor);
    if (submitRes < 0) {
        this->failureSet_->Insert(task.owner);
        task.done(false);
        return;
    }

    struct hf3fs_cqe cqe;
    int waitRes = hf3fs_wait_for_ios(&usrbio_.readIor,
                                    &cqe,
                                    1,
                                    1,
                                    nullptr);

    if (waitRes <= 0) {
        this->failureSet_->Insert(task.owner);
        task.done(false);
        return;
    }

    if (cqe.result < 0) {
        this->failureSet_->Insert(task.owner);
        task.done(false);
        return;
    }

    if (hostPtr > 0 && cqe.result > 0) {
        memcpy_fast((void*)hostPtr, usrbio_.readIov.base, cqe.result);
    }

    this->devPool_.Push(std::move(task));
}

TransQueue::UsrbioResources& TransQueue::GetThreadUsrbioResources()
{
    auto& usrbioPtr = g_threadUsrbioResources;
    if (usrbioPtr == nullptr) {
        usrbioPtr = new UsrbioResources();
        auto usrbio_ = static_cast<UsrbioResources*>(usrbioPtr);

        int res = hf3fs_iovcreate(&usrbio_->readIov,
                                  this->mountPoint_.c_str(),
                                  IOV_SIZE,
                                  0,
                                  -1);
        if (res < 0) {
            delete usrbio_;
            usrbioPtr = nullptr;
            throw std::runtime_error("Failed to create read Iov");
        }

        res = hf3fs_iovcreate(&usrbio_->writeIov,
                              this->mountPoint_.c_str(),
                              IOV_SIZE,
                              0,
                              -1);
        if (res < 0) {
            hf3fs_iovdestroy(&usrbio_->readIov);
            delete usrbio_;
            usrbioPtr = nullptr;
            throw std::runtime_error("Failed to create write Iov");
        }

        res = hf3fs_iorcreate4(&usrbio_->readIor,
                               this->mountPoint_.c_str(),
                               IOR_ENTRIES,
                               true,
                               IO_DEPTH,
                               0,
                               -1,
                               0);
        if (res < 0) {
            hf3fs_iovdestroy(&usrbio_->readIov);
            hf3fs_iovdestroy(&usrbio_->writeIov);
            delete usrbio_;
            usrbioPtr = nullptr;
            throw std::runtime_error("Failed to create read Ior");
        }

        res = hf3fs_iorcreate4(&usrbio_->writeIor,
                               this->mountPoint_.c_str(),
                               IOR_ENTRIES,
                               false,
                               IO_DEPTH,
                               0,
                               -1,
                               0);
        if (res < 0) {
            hf3fs_iovdestroy(&usrbio_->readIov);
            hf3fs_iovdestroy(&usrbio_->writeIov);
            hf3fs_iordestroy(&usrbio_->readIor);
            delete usrbio_;
            usrbioPtr = nullptr;
            throw std::runtime_error("Failed to create write Ior");
        }

        usrbio_->initialized = true;
    }

    return *static_cast<UsrbioResources*>(usrbioPtr);
}

Status TransQueue::InitUsrbio()
{
    int res = hf3fs_iovcreate(&usrbio_.readIov,
                              this->mountPoint_.c_str(),
                              IOV_SIZE,
                              0,
                              -1);

    if (res < 0) {
        return Status::Error();
    }

    res = hf3fs_iovcreate(&usrbio_.writeIov,
                          this->mountPoint_.c_str(),
                          IOV_SIZE,
                          0,
                          -1);

    if (res < 0) {
        hf3fs_iovdestroy(&usrbio_.readIov);
        return Status::Error();
    }

    res = hf3fs_iorcreate4(&usrbio_.readIor,
                           this->mountPoint_.c_str(),
                           IOR_ENTRIES,
                           true,
                           IO_DEPTH,
                           0,
                           -1,
                           0);

    if (res < 0) {
        hf3fs_iovdestroy(&usrbio_.readIov);
        hf3fs_iovdestroy(&usrbio_.writeIov);
        return Status::Error();
    }

    res = hf3fs_iorcreate4(&usrbio_.writeIor,
                           this->mountPoint_.c_str(),
                           IOR_ENTRIES,
                           false,
                           IO_DEPTH,
                           0,
                           -1,
                           0);

    if (res < 0) {
        hf3fs_iovdestroy(&usrbio_.readIov);
        hf3fs_iovdestroy(&usrbio_.writeIov);
        hf3fs_iordestroy(&usrbio_.readIor);
        return Status::Error();
    }

    usrbio_.initialized = true;
    return Status::OK();
}

Status TransQueue::CleanupUsrbio()
{
    if (!usrbio_.initialized) {
        return Status::OK();
    }

    hf3fs_iordestroy(&usrbio_.readIor);
    hf3fs_iordestroy(&usrbio_.writeIor);
    hf3fs_iovdestroy(&usrbio_.readIov);
    hf3fs_iovdestroy(&usrbio_.writeIov);

    usrbio_.initialized = false;
    return Status::OK();
}

Status TransQueue::Setup(const int32_t deviceId, const size_t streamNumber, const size_t blockSize,
                         const size_t ioSize, const bool ioDirect, const size_t bufferNumber,
                         const SpaceLayout* layout, TaskSet* failureSet_, const std::string& mountPoint)
{
    Trans::Device device;
    auto ts = device.Setup(deviceId);
    if (ts.Failure()) {
        return Status::Error();
    }

    buffer_ = device.MakeBuffer();
    stream_ = device.MakeStream();
    if (!buffer_ || !stream_) {
        return Status::Error();
    }

    ts = buffer_->MakeHostBuffers(blockSize, bufferNumber);
    if (ts.Failure()) {
        return Status::Error();
    }

    auto success =
        this->devPool_.SetWorkerFn([this](auto t, auto) { this->DeviceWorker(std::move(t)); })
            .Run();
    if (!success) {
        return Status::Error();
    }

    success = this->filePool_.SetWorkerFn([this](auto t, auto) { this->FileWorker(std::move(t)); })
                  .SetNWorker(streamNumber)
                  .Run();
    if (!success) {
        return Status::Error();
    }

    this->layout_ = layout;
    this->mountPoint_ = mountPoint;
    this->ioSize_ = ioSize;
    this->ioDirect_ = ioDirect;
    this->failureSet_ = failureSet_;

    ts = InitUsrbio();
    if (ts.Failure()) {
        return ts;
    }

    return Status::OK();
}

void TransQueue::Dispatch(TaskPtr task, WaiterPtr waiter)
{
    if (task->type == TransTask::Type::DUMP) {
        this->DispatchDump(task, waiter);
        return;
    }

    task->ForEachGroup(
        [task, waiter, this](const std::string& block, std::vector<uintptr_t>& shards) {
            BlockTask blockTask;
            blockTask.owner = task->id;
            blockTask.block = block;
            blockTask.type = task->type;
            auto bufferSize = this->ioSize_ * shards.size();
            std::swap(blockTask.shards, shards);
            blockTask.buffer = buffer_->GetHostBuffer(bufferSize);

            blockTask.done = [task, waiter, ioSize = this->ioSize_](bool success) {
                if (!success) {
                    waiter->Done(nullptr);
                } else {
                    waiter->Done([task, ioSize] { UC_DEBUG("{}", task->Epilog(ioSize)); });
                }
            };

            if (task->type == TransTask::Type::DUMP) {
                this->devPool_.Push(std::move(blockTask));
            } else {
                this->filePool_.Push(std::move(blockTask));
            }
        });
}

void TransQueue::DispatchDump(TaskPtr task, WaiterPtr waiter)
{
    std::vector<BlockTask> blocks;
    blocks.reserve(task->GroupNumber());

    task->ForEachGroup(
        [task, waiter, &blocks, this](const std::string& block, std::vector<uintptr_t>& shards) {
            BlockTask blockTask;
            blockTask.owner = task->id;
            blockTask.block = block;
            blockTask.type = task->type;
            auto bufferSize = this->ioSize_ * shards.size();
            blockTask.buffer = buffer_->GetHostBuffer(bufferSize);
            std::swap(blockTask.shards, shards);

            blockTask.done = [task, waiter, ioSize = this->ioSize_](bool success) {
                if (!success) {
                    waiter->Done(nullptr);
                } else {
                    waiter->Done([task, ioSize] { UC_DEBUG("{}", task->Epilog(ioSize)); });
                }
            };

            auto device = (void**)blockTask.shards.data();
            auto host = blockTask.buffer.get();

            stream_->DeviceToHostAsync(device, host, this->ioSize_, blockTask.shards.size());
            blocks.push_back(std::move(blockTask));
        });

    auto s = stream_->Synchronized();
    if (s.Failure()) {
        this->failureSet_->Insert(task->id);
    }

    for (auto&& block : blocks) {
        if (s.Failure()) {
            waiter->Done(nullptr);
            return;
        }

        this->filePool_.Push(std::move(block));
    }
}

} // namespace UC
