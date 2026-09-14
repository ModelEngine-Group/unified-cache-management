#ifndef UCM_CACHE_EARLY_STATS_H
#define UCM_CACHE_EARLY_STATS_H

#include "logger/logger.h"
#include "time/stopwatch.h"

namespace UC::CacheStore {
// One record per operation, including failures; no clock or log in shard loops.
struct EarlyStats {
    const char* operation;
    StopWatch elapsed{};
    uint64_t handle{0};
    bool success{false};
    double leaseMs{0}, pinMs{0}, lookupMs{0}, submitMs{0}, waitMs{0}, firstIoMs{-1};
    size_t shards{0}, lookups{0}, bucketLocks{0}, reusedBlocks{0}, indexBuilds{0}, matched{0};
    struct Phase {
        double& milliseconds;
        StopWatch timer{};
        ~Phase() { milliseconds += timer.Elapsed().count() * 1e3; }
    };
    static double Milliseconds(const StopWatch& timer) { return timer.Elapsed().count() * 1e3; }
    ~EarlyStats()
    {
        UC_INFO("UCM_EARLY stage={} reservation={} success={} total_ms={:.3f} lease_ms={:.3f} "
                "pin_ms={:.3f} l2_lookup_ms={:.3f} submit_ms={:.3f} wait_ms={:.3f} first_io_ms={:.3f} "
                "shards={} l2_calls={} bucket_locks={} reused_blocks={} index_builds={} handles={}",
                operation, handle, success, Milliseconds(elapsed), leaseMs, pinMs, lookupMs,
                submitMs, waitMs, firstIoMs, shards, lookups, bucketLocks, reusedBlocks,
                indexBuilds, matched);
    }
};
}  // namespace UC::CacheStore
#endif
