#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>
#include <torch/torch.h>

struct CopyInfo {
    uint32_t layerId;
    int32_t indexInBatch;
    torch::Tensor srcTensor;
}

class ThreadSafeQueue {
public:
    ThreadSafeQueue();
    ~ThreadSafeQueue() = default;

    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    void push(CopyInfo value);
    CopyInfo pop();
    size_t size() const;
    bool empty() const;
    void stop();
    void clear();

private:
    mutable std::mutex m_mutex;
    std::condition_variable m_condVar;
    std::queue<CopyInfo> m_queue;
    std::atomic<bool> m_stopped;
};

#endif