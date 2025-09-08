#ifndef KVSTAR_RETRIEVE_CLIB_TASK_RESULT_H
#define KVSTAR_RETRIEVE_CLIB_TASK_RESULT_H

#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <cstdint>
#include "domain/retrieve_task/task_status.h"


namespace KVStar {
struct TaskResult {
    std::atomic<TaskStatus> status{TaskStatus::PENDING};
    std::vector<int64_t> topkIndices;
    std::string errorMessage;
    std::mutex mtx;
    TaskResult() = default;
};

} // namespace KVStar

#endif
