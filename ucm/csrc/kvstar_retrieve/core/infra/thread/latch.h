#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_LATCH_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_LATCH_H

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace KVStar {
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