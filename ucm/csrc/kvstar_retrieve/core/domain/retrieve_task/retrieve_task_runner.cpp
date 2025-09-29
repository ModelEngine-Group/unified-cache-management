#include "retrieve_task_runner.h"
#include <functional>
#include <map>
#include <thread>   // 包含 std::this_thread::sleep_for
#include <chrono>   // 包含 std::chrono::seconds, std::chrono::milliseconds

#include "logger/logger.h"
#include "memory/memory.h"
#include "template/singleton.h"
#include "simd_compute_kernel.h"

namespace KVStar {

Status RetrieveTaskRunner::Run(const RetrieveTask& task, TaskResult& result) {
    try {
        // 任务对象中的数据是纯粹的PlainTensor, 不占GIL
        KVSTAR_DEBUG("Task {} starting pure C++ computation.", task.allocTaskId);

        // 直接将任务对象传给kernel
        KVStar::Execute(task, result);

        KVSTAR_DEBUG("Task {} pure C++ computation finished successfully.", task.allocTaskId);


    } catch (const std::exception& e) {
        // 错误处理
        // 打印错误详细信息
        KVSTAR_ERROR("Task {} failed during computation in Runner. Error: {}", task.allocTaskId, e.what());

        // 更新TaskResult的状态, 写入具体错误情况
        {
            std::lock_guard<std::mutex> lock(result.mtx);
            result.errorMessage = e.what();
            // 原子写更改任务状态
            result.status.store(TaskStatus::FAILURE, std::memory_order_release);
        }


    }

    return Status::OK();
}

}