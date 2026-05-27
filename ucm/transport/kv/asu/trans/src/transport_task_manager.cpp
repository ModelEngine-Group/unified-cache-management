#include "transport_task_manager.h"
#include "connection_internal.h"
#include "logger.h"

namespace UC::ASU {

bool TransportTaskContext::Done() const
{
    auto s = state.load(std::memory_order_acquire);
    return s == TransportTaskState::COMPLETED || s == TransportTaskState::FAILED ||
           s == TransportTaskState::CANCELED;
}

void TransportTaskManager::Shutdown()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [id, ctx] : tasks_) {
        (void)id;
        auto expected = ctx->state.load(std::memory_order_acquire);
        while (expected == TransportTaskState::PENDING ||
               expected == TransportTaskState::INFLIGHT) {
            ctx->finalStatus =
                Status::Error(StatusCode::CANCELED, "transport shutdown canceled task");
            if (ctx->state.compare_exchange_weak(expected, TransportTaskState::CANCELED,
                                                 std::memory_order_acq_rel)) {
                ctx->cv.notify_all();
                break;
            }
        }
    }
}

}  // namespace UC::ASU