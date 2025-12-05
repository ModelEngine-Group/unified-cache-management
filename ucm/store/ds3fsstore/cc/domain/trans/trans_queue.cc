#include "trans_queue.h"
#include <cstring>
#include <fcntl.h>
#include <hf3fs_usrbio.h>
#include <thread>
#include <unistd.h>
#include "file/file.h"
#include "logger/logger.h"
#include "trans/device.h"

namespace UC {

thread_local void* g_threadUsrbioResources = nullptr;

TransQueue::~TransQueue()
{
    fdHandlePool_.ClearAll([](const FdHandle& handle) {
        if (handle.regFd >= 0) { hf3fs_unreg_fd(handle.regFd); }
        if (handle.fd >= 0) { close(handle.fd); }
    });
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
        if (s.Success()) { this->filePool_.Push(std::move(task)); }
    }

    if (s.Failure()) { this->failureSet_->Insert(task.owner); }

    done(s.Success());
}

Status TransQueue::OpenFile(const std::string& path, bool isWrite, FdHandle& fdHandle)
{
    int openFlags = isWrite ? (O_WRONLY | O_CREAT | O_TRUNC) : O_RDONLY;
    int openMode = isWrite ? 0644 : 0;

    int fd = open(path.c_str(), openFlags, openMode);
    if (fd < 0) { return Status::Error(); }

    if (!hf3fs_is_hf3fs(fd)) {
        close(fd);
        return Status::Error();
    }

    int regRes = hf3fs_reg_fd(fd, 0);
    if (regRes > 0) {
        close(fd);
        return Status::Error();
    }

    fdHandle.fd = fd;
    fdHandle.regFd = regRes;
    return Status::OK();
}

Status TransQueue::DoWrite(const BlockTask& task, const FdHandle& fdHandle, UsrbioResources& usrbio)
{
    auto hostPtr = (uintptr_t)task.buffer.get();
    auto length = this->ioSize_ * task.shards.size();
    int regFd = fdHandle.regFd;

    if (hostPtr > 0 && length > 0) {
        if (length > usrbio.writeIov.size) { return Status::Error(); }
        memcpy(usrbio.writeIov.base, (void*)hostPtr, length);
    }

    int prepRes = hf3fs_prep_io(&usrbio.writeIor, &usrbio.writeIov, false, usrbio.writeIov.base,
                                regFd, 0, length, (const void*)(uintptr_t)task.owner);

    if (prepRes < 0) { return Status::Error(); }

    int submitRes = hf3fs_submit_ios(&usrbio.writeIor);
    if (submitRes < 0) { return Status::Error(); }

    struct hf3fs_cqe cqe;
    int waitRes = hf3fs_wait_for_ios(&usrbio.writeIor, &cqe, 1, 1, nullptr);

    if (waitRes <= 0) { return Status::Error(); }

    if (cqe.result < 0) { return Status::Error(); }

    this->layout_->Commit(task.block, true);
    return Status::OK();
}

Status TransQueue::DoRead(const BlockTask& task, const FdHandle& fdHandle, UsrbioResources& usrbio)
{
    auto hostPtr = (uintptr_t)task.buffer.get();
    auto length = this->ioSize_ * task.shards.size();
    int regFd = fdHandle.regFd;

    if (length > usrbio.readIov.size) { return Status::Error(); }

    int prepRes = hf3fs_prep_io(&usrbio.readIor, &usrbio.readIov, true, usrbio.readIov.base, regFd,
                                0, length, (const void*)(uintptr_t)task.owner);

    if (prepRes < 0) { return Status::Error(); }

    int submitRes = hf3fs_submit_ios(&usrbio.readIor);
    if (submitRes < 0) { return Status::Error(); }

    struct hf3fs_cqe cqe;
    int waitRes = hf3fs_wait_for_ios(&usrbio.readIor, &cqe, 1, 1, nullptr);

    if (waitRes <= 0) { return Status::Error(); }

    if (cqe.result < 0) { return Status::Error(); }

    if (hostPtr > 0 && cqe.result > 0) { memcpy((void*)hostPtr, usrbio.readIov.base, cqe.result); }

    return Status::OK();
}

void TransQueue::FileWorker(BlockTask&& task)
{
    if (this->failureSet_->Contains(task.owner)) {
        task.done(false);
        return;
    }

    UsrbioResources& usrbio_ = GetThreadUsrbioResources();
    bool isDump = task.type == TransTask::Type::DUMP;
    const auto& path = this->layout_->DataFilePath(task.block, isDump);

    FdHandle fdHandle{};
    auto status =
        fdHandlePool_.Get(path, fdHandle, [&path, isDump, this](FdHandle& handle) -> Status {
            return OpenFile(path, isDump, handle);
        });

    if (status.Failure()) {
        this->failureSet_->Insert(task.owner);
        task.done(false);
        return;
    }

    Status result = isDump ? DoWrite(task, fdHandle, usrbio_) : DoRead(task, fdHandle, usrbio_);

    if (result.Failure()) {
        this->failureSet_->Insert(task.owner);
        task.done(false);
        return;
    }

    if (isDump) {
        task.done(true);
    } else {
        this->devPool_.Push(std::move(task));
    }
}

TransQueue::UsrbioResources& TransQueue::GetThreadUsrbioResources()
{
    auto& usrbioPtr = g_threadUsrbioResources;
    if (usrbioPtr == nullptr) {
        usrbioPtr = new UsrbioResources();
        auto usrbio_ = static_cast<UsrbioResources*>(usrbioPtr);

        int res = hf3fs_iovcreate(&usrbio_->readIov, this->mountPoint_.c_str(), IOV_SIZE, 0, -1);
        if (res < 0) {
            delete usrbio_;
            usrbioPtr = nullptr;
            throw std::runtime_error("Failed to create read Iov");
        }

        res = hf3fs_iovcreate(&usrbio_->writeIov, this->mountPoint_.c_str(), IOV_SIZE, 0, -1);
        if (res < 0) {
            hf3fs_iovdestroy(&usrbio_->readIov);
            delete usrbio_;
            usrbioPtr = nullptr;
            throw std::runtime_error("Failed to create write Iov");
        }

        res = hf3fs_iorcreate4(&usrbio_->readIor, this->mountPoint_.c_str(), IOR_ENTRIES, true,
                               IO_DEPTH, 0, -1, 0);
        if (res < 0) {
            hf3fs_iovdestroy(&usrbio_->readIov);
            hf3fs_iovdestroy(&usrbio_->writeIov);
            delete usrbio_;
            usrbioPtr = nullptr;
            throw std::runtime_error("Failed to create read Ior");
        }

        res = hf3fs_iorcreate4(&usrbio_->writeIor, this->mountPoint_.c_str(), IOR_ENTRIES, false,
                               IO_DEPTH, 0, -1, 0);
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
    int res = hf3fs_iovcreate(&usrbio_.readIov, this->mountPoint_.c_str(), IOV_SIZE, 0, -1);

    if (res < 0) { return Status::Error(); }

    res = hf3fs_iovcreate(&usrbio_.writeIov, this->mountPoint_.c_str(), IOV_SIZE, 0, -1);

    if (res < 0) {
        hf3fs_iovdestroy(&usrbio_.readIov);
        return Status::Error();
    }

    res = hf3fs_iorcreate4(&usrbio_.readIor, this->mountPoint_.c_str(), IOR_ENTRIES, true, IO_DEPTH,
                           0, -1, 0);

    if (res < 0) {
        hf3fs_iovdestroy(&usrbio_.readIov);
        hf3fs_iovdestroy(&usrbio_.writeIov);
        return Status::Error();
    }

    res = hf3fs_iorcreate4(&usrbio_.writeIor, this->mountPoint_.c_str(), IOR_ENTRIES, false,
                           IO_DEPTH, 0, -1, 0);

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
    if (!usrbio_.initialized) { return Status::OK(); }

    hf3fs_iordestroy(&usrbio_.readIor);
    hf3fs_iordestroy(&usrbio_.writeIor);
    hf3fs_iovdestroy(&usrbio_.readIov);
    hf3fs_iovdestroy(&usrbio_.writeIov);

    usrbio_.initialized = false;
    return Status::OK();
}

Status TransQueue::Setup(const int32_t deviceId, const size_t streamNumber, const size_t blockSize,
                         const size_t ioSize, const bool ioDirect, const size_t bufferNumber,
                         const SpaceLayout* layout, TaskSet* failureSet_,
                         const std::string& mountPoint)
{
    Trans::Device device;
    auto ts = device.Setup(deviceId);
    if (ts.Failure()) { return Status::Error(); }

    buffer_ = device.MakeBuffer();
    stream_ = device.MakeStream();
    if (!buffer_ || !stream_) { return Status::Error(); }

    ts = buffer_->MakeHostBuffers(blockSize, bufferNumber);
    if (ts.Failure()) { return Status::Error(); }

    auto success =
        this->devPool_.SetWorkerFn([this](auto t, auto) { this->DeviceWorker(std::move(t)); })
            .Run();
    if (!success) { return Status::Error(); }

    success = this->filePool_.SetWorkerFn([this](auto t, auto) { this->FileWorker(std::move(t)); })
                  .SetNWorker(streamNumber)
                  .Run();
    if (!success) { return Status::Error(); }

    this->layout_ = layout;
    this->mountPoint_ = mountPoint;
    this->ioSize_ = ioSize;
    this->ioDirect_ = ioDirect;
    this->failureSet_ = failureSet_;

    ts = InitUsrbio();
    if (ts.Failure()) { return ts; }

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
    if (s.Failure()) { this->failureSet_->Insert(task->id); }

    for (auto&& block : blocks) {
        if (s.Failure()) {
            waiter->Done(nullptr);
            return;
        }

        this->filePool_.Push(std::move(block));
    }
}

} // namespace UC
