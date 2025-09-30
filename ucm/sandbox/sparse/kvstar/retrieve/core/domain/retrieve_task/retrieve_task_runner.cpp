#include "retrieve_task_runner.h"
#include <functional>
#include <map>
#include <thread>
#include <chrono>

#include "logger/logger.h"
#include "memory/memory.h"
#include "template/singleton.h"
#include "simd_compute_kernel.h"

namespace KVStar {

Status RetrieveTaskRunner::Run(const RetrieveTask& task, TaskResult& result) {
    try {
        KVSTAR_DEBUG("Task {} starting pure C++ computation.", task.allocTaskId);

        KVStar::Execute(task, result);

        KVSTAR_DEBUG("Task {} pure C++ computation finished successfully.", task.allocTaskId);


    } catch (const std::exception& e) {
        KVSTAR_ERROR("Task {} failed during computation in Runner. Error: {}", task.allocTaskId, e.what());

        {
            std::lock_guard<std::mutex> lock(result.mtx);
            result.errorMessage = e.what();
            result.status.store(TaskStatus::FAILURE, std::memory_order_release);
        }


    }

    return Status::OK();
}

}