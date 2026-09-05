#include "trans_task_manager.h"
#include <utility>
#include "utils/trans_task_utils.h"

namespace kv {

TransportTask::TransportTask() : subBatchContexts(std::make_shared<TransportSubBatchList>()) {}

bool TransportTask::Done() const
{
    return state.load(std::memory_order_acquire) == TransportTaskState::COMPLETED;
}

bool TransportTask::NotifyCompletion(TaskResult result)
{
    if (!onComplete || completionNotified.exchange(true, std::memory_order_acq_rel)) {
        return false;
    }
    onComplete(std::move(result));
    return true;
}

Status TransportTask::BuildFinalStatus() const
{
    for (const auto& subBatchContext : *subBatchContexts) {
        if (!subBatchContext.status.ok()) {
            return Status::Error(StatusCode::PARTIAL_FAILED, "transport task partially failed");
        }
    }

    return Status::OK();
}

void TransportTask::InitializeRemainingSubBatchCount()
{
    remainingSubBatchCount = 0;
    for (const auto& subBatchContext : *subBatchContexts) {
        if (subBatchContext.state == TransportSubBatchState::PENDING) { ++remainingSubBatchCount; }
    }
}

void TransportTask::TryFinalizeFromSubBatches()
{
    if (subBatchContexts->empty()) {
        finalStatus = Status::Error(StatusCode::PARTIAL_FAILED, "transport task partially failed");
        state.store(TransportTaskState::COMPLETED, std::memory_order_release);
        return;
    }

    if (remainingSubBatchCount != 0) { return; }

    finalStatus = BuildFinalStatus();
    state.store(TransportTaskState::COMPLETED, std::memory_order_release);
}

void TransportTaskManager::NotifyCompletion(const TransportTaskPtr& task)
{
    TaskResult result;
    BuildResult(*task, result);
    (void)task->NotifyCompletion(std::move(result));
    (void)Remove(task->taskId);
}

void TransportTaskManager::BuildResult(const TransportTask& task, TaskResult& result)
{
    result.status = task.finalStatus;
    result.entryStatus = task.entryStatus;
    if (!task.subBatchContexts->empty()) {
        std::size_t resultIndex = 0;
        for (const auto& subBatchContext : *task.subBatchContexts) {
            for (const auto& status : subBatchContext.entryStatus) {
                if (resultIndex >= result.entryStatus.size()) { break; }
                result.entryStatus[resultIndex++] = status;
            }
        }
    }

    result.queryResult.reset();
    if (task.opType == AsuOpType::QUERY) {
        result.queryResult = BuildQueryResultFromEntryStatus(result.entryStatus);
    }
}

}  // namespace kv
