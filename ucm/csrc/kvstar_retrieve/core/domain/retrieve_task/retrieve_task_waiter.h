#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_WAITER_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_WAITER_H

#include <spdlog/fmt/fmt.h>
#include <spdlog/stopwatch.h>
#include "logger/logger.h"
#include "thread/latch.h"

namespace KVStar {

class RetrieveTaskWaiter : public Latch {
public:
    RetrieveTaskWaiter(const size_t taskId, const size_t waitCounter)
        : Latch{waitCounter}, _taskId{taskId}, _waitCounter{waitCounter}
    {
    }

    void Done()
    {
        if (Latch::Done() == 0) {
            KVSTAR_DEBUG("Task({}, {}) finished, elapsed {:.06f}s", this->_taskId, this->_waitCounter, this->_sw.elapsed().count());
            this->Notify();
        }
    }

private:
    size_t _taskId;
    size_t _waitCounter;
    spdlog::stopwatch _sw;

};

}

#endif //UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_WAITER_H
