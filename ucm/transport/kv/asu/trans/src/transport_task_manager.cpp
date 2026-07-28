#include "transport_task_manager.h"

namespace UC::ASU {

bool TransportTaskContext::Done() const
{
    auto s = state.load(std::memory_order_acquire);
    return s == TransportTaskState::COMPLETED || s == TransportTaskState::CANCELED;
}

bool TransportTaskContext::NotifyCompletion(TaskResult result)
{
    if (!onComplete || completionNotified.exchange(true, std::memory_order_acq_rel)) {
        return false;
    }
    onComplete(std::move(result));
    return true;
}

}  // namespace UC::ASU
