#ifndef UCM_EARLY_READ_COMPLETION_H
#define UCM_EARLY_READ_COMPLETION_H
#include <chrono>
#include <thread>
#include "logger/logger.h"
#include "ucmstore_v1.h"

namespace UC::CacheStore {
// Common completion boundary for reservation batches and pre-acquired native
// loads. Check failure is not completion evidence. The caller retains pins and
// leases throughout; only the backend which submitted the operation reaps it.
inline Status DrainEarlyRead(StoreV1& backend, Detail::TaskHandle handle)
{
    bool warned = false;
    for (;;) {
        auto check = backend.Check(handle);
        if (check && check.Value()) { return backend.Wait(handle); }
        if (!check && !warned) {
            UC_ERROR("Early read({}) completion unknown ({}); retaining I/O protection.",
                     handle, check.Error());
            warned = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

template <typename Indices, typename PinAt>
Status CompleteEarlyRead(StoreV1& backend, Detail::TaskHandle handle,
                         const Indices& indices, PinAt pinAt)
{
    auto status = DrainEarlyRead(backend, handle);
    for (auto index : indices) {
        auto& pin = pinAt(index);
        if (status.Success()) { pin.MarkReady(); }
        else { pin.MarkFailed(status); }
    }
    return status;
}
}
#endif
