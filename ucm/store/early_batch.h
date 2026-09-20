#ifndef UCM_EARLY_BATCH_H
#define UCM_EARLY_BATCH_H

#include <functional>
#include "ucmstore_v1.h"

namespace UC::Early {
// A capability of an existing reservation, not another Store or I/O executor.
// Step never waits for unfinished I/O. Run retains the synchronous API.
struct Progress {
    bool waitingCapacity{false};
    uint32_t waitEpoch{0};
    bool done{false};
    bool advanced{false};
    bool canSubmit{true};
    bool discarded{false};  // explicit suffix retirement, after drain; not an I/O failure
    size_t inflightBytes{0};
    size_t remainingL2Work{0};
    size_t completedReadBytes{0};
    PrefetchResult result{};
};

class Batch {
protected:
    std::function<bool()> healthy_ = [] { return true; };
public:
    const size_t shardBytes, windowBytes, protectionLimitBytes;
    Batch(size_t shard, size_t window, size_t protectionLimit)
        : shardBytes(shard), windowBytes(window), protectionLimitBytes(protectionLimit) {}
    virtual ~Batch() = default;
    virtual Expected<Progress> Step(size_t availableBytes) = 0;
    virtual Progress Snapshot() = 0;
    // Deferred query owns identities/lease only; execution revalidates L1.
    virtual Expected<Progress> Activate() { return Status::Error("not deferred"); }
    virtual void Limit(size_t) {}  // before activation, intrinsic oversize only
    virtual void Cutoff() {}  // exceptional unsafe/impossible wait, never normal NoSpace
    virtual void Discard() = 0;  // stop new suffix I/O; Step still collects submitted I/O
    virtual Expected<PrefetchResult> Run() = 0;
    virtual size_t ProtectedBytes() const = 0;
    // Installed by the existing health wrapper before publication to Python.
    void SetHealthGuard(std::function<bool()> healthy) { healthy_ = std::move(healthy); }
};
}  // namespace UC::Early
#endif
