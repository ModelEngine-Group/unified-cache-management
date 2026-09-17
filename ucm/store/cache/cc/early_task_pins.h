#ifndef UCM_EARLY_TASK_PINS_H
#define UCM_EARLY_TASK_PINS_H
#include "trans_buffer.h"

namespace UC::CacheStore {
// Unsubmitted/undispatched allocation owners must not leave a LOADING node.
// The queues move accepted handles out; their existing I/O lifetime takes over.
class EarlyTaskPins : public std::vector<TransBuffer::Handle> {
public:
    using std::vector<TransBuffer::Handle>::vector;
    using std::vector<TransBuffer::Handle>::operator=;
    EarlyTaskPins() = default;
    EarlyTaskPins(EarlyTaskPins&&) = default;
    EarlyTaskPins(const EarlyTaskPins&) = delete;
    ~EarlyTaskPins()
    {
        for (auto& pin : *this) {
            if (pin && pin.Owner() && pin.GetState() == TransBuffer::State::LOADING) {
                pin.MarkFailed(Status::NotFound());
            }
        }
    }
};
}
#endif
