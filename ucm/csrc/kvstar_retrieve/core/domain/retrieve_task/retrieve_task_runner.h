#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_RUNNER_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_RUNNER_H

#include "status/status.h"
#include "retrieve_task.h"
#include "task_result.h"


namespace KVStar {

class RetrieveTaskRunner {
public:
    RetrieveTaskRunner(){}
    Status Run(const RetrieveTask& task, TaskResult& result);
};


}


#endif //UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_RUNNER_H