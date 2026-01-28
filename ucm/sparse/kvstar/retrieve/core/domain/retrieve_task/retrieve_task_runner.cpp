#include "retrieve_task_runner.h"
#include <chrono>
#include <functional>
#include <map>
#include <thread>
#include "logger/logger.h"
#include "memory/memory.h"
#include "simd_compute_kernel.h"
#include "template/singleton.h"

namespace KVStar {

Status RetrieveTaskRunner::Run(const RetrieveTask& task, TaskResult& result)
{
    try {
        UC_DEBUG("Task {} starting pure C++ computation.", task.allocTaskId);

        KVStar::Execute(task, result);

        UC_DEBUG("Task {} pure C++ computation finished successfully.", task.allocTaskId);

    } catch (const std::exception& e) {
        UC_ERROR("Task {} failed during computation in Runner. Error: {}", task.allocTaskId,
                 e.what());

        {
            std::lock_guard<std::mutex> lock(result.mtx);
            result.errorMessage = e.what();
            result.status.store(TaskStatus::FAILURE, std::memory_order_release);
        }
    }

    return Status::OK();
}

}  // namespace KVStar