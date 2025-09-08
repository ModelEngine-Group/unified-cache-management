#ifndef KVSTAR_RETRIEVE_SIMD_COMPUTE_KERNEL_H
#define KVSTAR_RETRIEVE_SIMD_COMPUTE_KERNEL_H

#include "retrieve_task.h"
#include "task_result.h"

namespace KVStar {
/**
   检索计算的核心函数
   接收python无关(绕GIL)的任务描述, 进行检索
   结果保存到result, 后续根据任务ID可查
*/
// 类NPU API
void Execute(const RetrieveTask& task, TaskResult& result);

}


#endif //KVSTAR_RETRIEVE_SIMD_COMPUTE_KERNEL_H