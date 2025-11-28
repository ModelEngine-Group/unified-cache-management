#ifndef UNIFIEDCACHE_TRANS_QUEUE_USRBIO_H
#define UNIFIEDCACHE_TRANS_QUEUE_USRBIO_H

#include "space/space_layout.h"
#include "task/task_set.h"
#include "task/task_waiter.h"
#include "thread/thread_pool.h"
#include "trans/buffer.h"
#include "trans/stream.h"
#include "trans_task.h"
#include "handle_recorder.h"
#include <hf3fs_usrbio.h>

namespace UC {

class TransQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<TaskWaiter>;

    struct BlockTask {
        size_t owner;
        std::string block;
        TransTask::Type type;
        std::vector<uintptr_t> shards;
        std::shared_ptr<void> buffer;
        std::function<void(bool)> done;
    };

    struct FdHandle {
        int fd;
        int regFd;
    };

    static constexpr size_t IOV_SIZE = 1UL << 30;
    static constexpr int IOR_ENTRIES = 64;
    static constexpr int IO_DEPTH = 0;

    struct UsrbioResources {
        struct hf3fs_iov readIov;
        struct hf3fs_iov writeIov;
        struct hf3fs_ior readIor;
        struct hf3fs_ior writeIor;
        bool initialized{false};
    };

    void DeviceWorker(BlockTask&& task);
    void FileWorker(BlockTask&& task);
    UsrbioResources& GetThreadUsrbioResources();
    Status InitUsrbio();
    Status CleanupUsrbio();
    Status OpenFile(const std::string& path, bool isWrite, FdHandle& fdHandle);
    Status DoWrite(const BlockTask& task, const FdHandle& fdHandle, UsrbioResources& usrbio);
    Status DoRead(const BlockTask& task, const FdHandle& fdHandle, UsrbioResources& usrbio);

public:
    ~TransQueue();
    Status Setup(const int32_t deviceId, const size_t streamNumber, const size_t blockSize,
                 const size_t ioSize, const bool ioDirect, const size_t bufferNumber,
                 const SpaceLayout* layout, TaskSet* failureSet_, const std::string& mountPoint);

    void Dispatch(TaskPtr task, WaiterPtr waiter);
    void DispatchDump(TaskPtr task, WaiterPtr waiter);

private:
    std::unique_ptr<Trans::Buffer> buffer_{nullptr};
    std::unique_ptr<Trans::Stream> stream_{nullptr};
    const SpaceLayout* layout_;
    std::string mountPoint_;
    size_t ioSize_;
    bool ioDirect_;
    ThreadPool<BlockTask> devPool_;
    ThreadPool<BlockTask> filePool_;
    TaskSet* failureSet_;

    UsrbioResources usrbio_;
    HandlePool<std::string, FdHandle>& fdHandlePool_{HandlePool<std::string, FdHandle>::Instance()};
};

} // namespace UC

#endif
