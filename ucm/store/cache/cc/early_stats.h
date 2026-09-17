#ifndef UCM_CACHE_EARLY_STATS_H
#define UCM_CACHE_EARLY_STATS_H

#include "logger/logger.h"
#include "time/stopwatch.h"
#include <chrono>
#include <fmt/ranges.h>
#include <vector>

namespace UC::CacheStore {
// Optional DEBUG detail per operation; failure propagation is independent.
// No clock or log in shard loops, and disabled detail avoids formatting.
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
    int64_t leaseStartNs{0}, leaseEndNs{0}, pinStartNs{0}, pinEndNs{0}, lookupStartNs{0}, lookupEndNs{0};
    size_t inflightPeakBytes{0}, ioBatches{0};
    uint64_t handle{0};
    std::vector<uint64_t> loadReservations;
    bool success{false};
    double leaseMs{0}, pinMs{0}, lookupMs{0}, submitMs{0}, waitMs{0}, firstIoMs{-1};
    size_t shards{0}, lookups{0}, bucketLocks{0}, reusedBlocks{0}, indexBuilds{0}, matched{0}, blockLookups{0};
    size_t recalledBlocks{0}, hintChecks{0}, hintEntries{0};
    size_t joinedBlocks{0}, protectedBytes{0}, protectionLimitBytes{0};
    double joinMs{0};
    struct Phase {
        double& milliseconds;
        StopWatch timer{};
        int64_t* end{nullptr};
        explicit Phase(double& ms) : milliseconds(ms) {}
        Phase(double& ms, int64_t& first, int64_t& last) : milliseconds(ms), end(&last)
        { if (first == 0) { first = NowNs(); } }
        ~Phase()
        {
            milliseconds += timer.Elapsed().count() * 1e3;
            if (end) { *end = NowNs(); }
        }
    };
    static double Milliseconds(const StopWatch& timer) { return timer.Elapsed().count() * 1e3; }
    ~EarlyStats()
    {
        if (!UC::Logger::isEnabledFor(UC::Logger::Level::DEBUG)) { return; }
        UC_DEBUG_UNLIMITED("UCM_EARLY stage={} reservation={} start_ns={} end_ns={} success={} total_ms={:.3f} lease_ms={:.3f} "
                "pin_ms={:.3f} l2_lookup_ms={:.3f} submit_ms={:.3f} wait_ms={:.3f} first_io_ms={:.3f} "
                "shards={} l2_calls={} bucket_locks={} reused_blocks={} index_builds={} handles={} block_lookups={} "
                "recalled_blocks={} hint_checks={} hint_entries={} first_io_ns={} last_io_ns={} "
                "inflight_peak_bytes={} io_batches={} joined_blocks={} join_ms={:.3f} "
                "protected_reference_bytes={} physical_reservable_bytes={} lease_start_ns={} lease_end_ns={} "
                "pin_start_ns={} pin_end_ns={} l2_start_ns={} l2_end_ns={} load_reservations={}",
                operation, handle, startNs, NowNs(), success, Milliseconds(elapsed), leaseMs, pinMs, lookupMs,
                submitMs, waitMs, firstIoMs, shards, lookups, bucketLocks, reusedBlocks,
                indexBuilds, matched, blockLookups, recalledBlocks, hintChecks, hintEntries,
                firstIoNs, lastIoNs, inflightPeakBytes, ioBatches, joinedBlocks, joinMs,
                protectedBytes, protectionLimitBytes, leaseStartNs, leaseEndNs,
                pinStartNs, pinEndNs, lookupStartNs, lookupEndNs,
                fmt::join(loadReservations, ","));
    }
};
}  // namespace UC::CacheStore
#endif
