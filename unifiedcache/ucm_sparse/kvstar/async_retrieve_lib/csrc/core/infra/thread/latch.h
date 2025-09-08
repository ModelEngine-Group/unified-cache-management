#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_LATCH_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_LATCH_H

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace KVStar {

// “门闩 (Latch)”同步原语, 线程同步
// 初始化一个计数器：表示需要等待完成的事件/任务数量。
// 一个或多个线程可以调用 Wait() 方法来阻塞自己，直到计数器归零
// 工作线程在完成自己的任务后，调用 Done() 方法来使计数器减一
// 当最后一个任务完成，计数器变为0时，调用 Notify()来唤醒所有正在 Wait() 的线程
class Latch {
public:
    explicit Latch(const size_t expected = 0) : _counter{expected} {}
    void Up() { ++this->_counter; }
    size_t Done() { return --this->_counter; }
    void Notify() { this->_cv.notify_all(); }
    void Wait()
    {
        std::unique_lock<std::mutex> lk(this->_mutex);
        if (this->_counter == 0) { return; }
        this->_cv.wait(lk, [this] { return this->_counter == 0; });
    }

private:
    std::mutex _mutex;
    std::condition_variable _cv;
    std::atomic<size_t> _counter;
};

}



#endif //UCM_SPARSE_KVSTAR_RETRIEVE_LATCH_H