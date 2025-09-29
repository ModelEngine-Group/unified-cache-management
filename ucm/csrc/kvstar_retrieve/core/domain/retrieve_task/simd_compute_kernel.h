#ifndef KVSTAR_RETRIEVE_SIMD_COMPUTE_KERNEL_H
#define KVSTAR_RETRIEVE_SIMD_COMPUTE_KERNEL_H

#include "retrieve_task.h"
#include "task_result.h"

namespace KVStar {

void Execute(const RetrieveTask& task, TaskResult& result);

}


#endif //KVSTAR_RETRIEVE_SIMD_COMPUTE_KERNEL_H