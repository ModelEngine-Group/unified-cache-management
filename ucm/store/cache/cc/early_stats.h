#ifndef UCM_CACHE_EARLY_STATS_H
#define UCM_CACHE_EARLY_STATS_H

#include "logger/logger.h"
#include "time/stopwatch.h"
#include <chrono>

namespace UC::CacheStore {
// One record per operation, including failures; no clock or log in shard loops.
struct EarlyStats {
    const char* operation;
    explicit EarlyStats(const char* name) : operation(name) {}
    StopWatch elapsed{};
    static int64_t NowNs()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    int64_t startNs{NowNs()};
    int64_t firstIoNs{0}, lastIoNs{0};
    size_t inflightPeakBytes{0}, ioBatches{0};
    uint64_t handle{0};
    bool success{false};
    double leaseMs{0}, pinMs{0}, lookupMs{0}, submitMs{0}, waitMs{0}, firstIoMs{-1};
    size_t shards{0}, lookups{0}, bucketLocks{0}, reusedBlocks{0}, indexBuilds{0}, matched{0}, blockLookups{0};
    size_t recalledBlocks{0}, hintChecks{0}, hintEntries{0};
    struct Phase {
        double& milliseconds;
        StopWatch timer{};
        ~Phase() { milliseconds += timer.Elapsed().count() * 1e3; }
    };
    static double Milliseconds(const StopWatch& timer) { return timer.Elapsed().count() * 1e3; }
    ~EarlyStats()
    {
        UC_INFO_UNLIMITED("UCM_EARLY stage={} reservation={} start_ns={} end_ns={} success={} total_ms={:.3f} lease_ms={:.3f} "
                "pin_ms={:.3f} l2_lookup_ms={:.3f} submit_ms={:.3f} wait_ms={:.3f} first_io_ms={:.3f} "
                "shards={} l2_calls={} bucket_locks={} reused_blocks={} index_builds={} handles={} block_lookups={} "
                "recalled_blocks={} hint_checks={} hint_entries={} first_io_ns={} last_io_ns={} "
                "inflight_peak_bytes={} io_batches={}",
                operation, handle, startNs, NowNs(), success, Milliseconds(elapsed), leaseMs, pinMs, lookupMs,
                submitMs, waitMs, firstIoMs, shards, lookups, bucketLocks, reusedBlocks,
                indexBuilds, matched, blockLookups, recalledBlocks, hintChecks, hintEntries,
                firstIoNs, lastIoNs, inflightPeakBytes, ioBatches);
    }
};
}  // namespace UC::CacheStore
#endif
