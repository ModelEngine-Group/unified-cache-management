#ifndef UCM_EARLY_BATCH_H
#define UCM_EARLY_BATCH_H

#include <functional>
#include "ucmstore_v1.h"

namespace UC::Early {
// A capability of an existing reservation, not another Store or I/O executor.
// Step never waits for unfinished I/O. Run retains the synchronous API.
struct Progress {
    bool done{false};
    bool advanced{false};
    size_t inflightBytes{0};
    PrefetchResult result{};
};

class Batch {
protected:
    std::function<bool()> healthy_ = [] { return true; };
public:
    const size_t shardBytes, windowBytes;
    Batch(size_t shard, size_t window) : shardBytes(shard), windowBytes(window) {}
    virtual ~Batch() = default;
    virtual Expected<Progress> Step(size_t availableBytes) = 0;
    virtual Expected<PrefetchResult> Run() = 0;
    // Installed by the existing health wrapper before publication to Python.
    void SetHealthGuard(std::function<bool()> healthy) { healthy_ = std::move(healthy); }
};
}  // namespace UC::Early
#endif
