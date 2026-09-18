/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#pragma once

#include <atomic>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>
#include "drampool_config.h"
#include "drampool_types.h"
#include "metrics_api.h"

namespace UC::DramPool {

// Queue length accountings backing the queue_request_size / queue_completion_size
// gauges. The SPSC queues expose no size query, and a gauge can only be written
// from a single thread (thread-local buffers, C5), so the producer and the
// consumer maintain these counters and each consumer thread reports the value.
inline std::atomic<std::uint64_t> g_requestQueueLen{0};
inline std::atomic<std::uint64_t> g_completionQueueLen{0};

// ---- metric names (grouped as in metrics_design.md §3.2) ----
// A. Request volume
inline constexpr char kDumpRequestsTotal[] = "drampool_dump_requests_total";
inline constexpr char kLoadRequestsTotal[] = "drampool_load_requests_total";
inline constexpr char kLookupRequestsTotal[] = "drampool_lookup_requests_total";
// B. Dump business
inline constexpr char kDumpNospaceFailuresTotal[] = "drampool_dump_nospace_failures_total";
inline constexpr char kDumpFailedEntriesTotal[] = "drampool_dump_failed_entries_total";
inline constexpr char kDumpPrepareDurationMs[] = "drampool_dump_prepare_duration_ms";
// C. Load business
inline constexpr char kLoadMissEntriesTotal[] = "drampool_load_miss_entries_total";
inline constexpr char kLoadPrepareDurationMs[] = "drampool_load_prepare_duration_ms";
// D. Lookup business
inline constexpr char kLookupMissEntriesTotal[] = "drampool_lookup_miss_entries_total";
inline constexpr char kLookupScanDurationMs[] = "drampool_lookup_scan_duration_ms";
// E. Transfer and response
inline constexpr char kDumpTransferDurationMs[] = "drampool_dump_transfer_duration_ms";
inline constexpr char kLoadTransferDurationMs[] = "drampool_load_transfer_duration_ms";
inline constexpr char kTransferFailuresTotal[] = "drampool_transfer_failures_total";
inline constexpr char kResponseRttMs[] = "drampool_response_rtt_ms";
inline constexpr char kResponseFailuresTotal[] = "drampool_response_failures_total";
inline constexpr char kSubmitFailuresTotal[] = "drampool_submit_failures_total";
// F. Resource usage
inline constexpr char kMetadataEntryCount[] = "drampool_metadata_entry_count";
inline constexpr char kFlagPoolUsageRatio[] = "drampool_flag_pool_usage_ratio";
// G. Metadata settlement duration
inline constexpr char kMetadataStoreendDurationMs[] = "drampool_metadata_storeend_duration_ms";
inline constexpr char kMetadataLoadendDurationMs[] = "drampool_metadata_loadend_duration_ms";
inline constexpr char kMetadataEvictGcDurationMs[] = "drampool_metadata_evict_gc_duration_ms";
// H. Queues and blocking
inline constexpr char kQueueRequestFullTotal[] = "drampool_queue_request_full_total";
inline constexpr char kQueueRequestEnqueueWaitMs[] = "drampool_queue_request_enqueue_wait_ms";
inline constexpr char kQueueRequestSize[] = "drampool_queue_request_size";
inline constexpr char kQueueCompletionFullTotal[] = "drampool_queue_completion_full_total";
inline constexpr char kQueueCompletionInflight[] = "drampool_queue_completion_inflight";
inline constexpr char kQueueCompletionSize[] = "drampool_queue_completion_size";
inline constexpr char kQueueResponseBufferRetryTotal[] =
    "drampool_queue_response_buffer_retry_total";
// I. Batch total duration
inline constexpr char kDumpBatchTotalDurationMs[] = "drampool_dump_batch_total_duration_ms";
inline constexpr char kLoadBatchTotalDurationMs[] = "drampool_load_batch_total_duration_ms";
inline constexpr char kLookupBatchTotalDurationMs[] = "drampool_lookup_batch_total_duration_ms";

// Dynamic per-slot-size gauge: drampool_buffer_pool_usage_ratio_<slot_size>.
// Names are built at runtime, so call sites keep the string-based UpdateStats
// overload; the only caller is the low-frequency GC report loop.
inline std::string BufferPoolUsageRatioName(std::uint64_t slotSize)
{
    return std::string("drampool_buffer_pool_usage_ratio_") + std::to_string(slotSize);
}

// RAII duration observer: measures with SteadyNowUs() and records the elapsed
// time in ms on scope exit. Takes a NAME_TO_METRIC_ID() reference so the metric
// id is resolved once per call site instead of a string lookup per observation.
// Call Disarm() on paths that must not be observed (e.g. failed preparations,
// see metrics_design.md §4.1).
class ScopedTimer {
public:
    explicit ScopedTimer(UC::Metrics::CachedMetric& metric)
        : metric_(metric), startUs_(SteadyNowUs())
    {
    }

    ~ScopedTimer()
    {
        if (armed_) {
            UC::Metrics::UpdateStats(metric_,
                                     static_cast<double>(SteadyNowUs() - startUs_) / 1000.0);
        }
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    void Disarm() { armed_ = false; }

private:
    UC::Metrics::CachedMetric& metric_;
    std::uint64_t startUs_;
    bool armed_{true};
};

// Hot-path metric updates resolve the metric id once per call site via the
// NAME_TO_METRIC_ID() cached reference (same pattern as DramStore), falling
// back to the string-based overload only for the dynamic per-slot-size gauges.

// Histogram bucket sets shared by metrics with the same latency envelope.
// The values mirror examples/metrics/metrics_configs.yaml, which is the single
// source of truth for metric names, types, and buckets (the Python side
// registers the same set via ucmmetrics.create_stats); keep both in sync.
inline constexpr double kMsBucketsSettlement[] = {0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500};
inline constexpr double kMsBucketsPrepare[] = {0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000};
inline constexpr double kMsBucketsTransfer[] = {0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000};
inline constexpr double kMsBucketsScan[] = {0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500};
inline constexpr double kMsBucketsLookupBatch[] = {0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, 1000};
inline constexpr double kMsBucketsGc[] = {1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000};

// One metric definition: name, type, and histogram buckets. It mirrors one
// entry of examples/metrics/metrics_configs.yaml and is consumed by both
// SetupDrampoolMetrics() (CreateStats registration) and MetricsReporter
// (Prometheus text rendering), so the daemon keeps a single registration
// source just like the Python side looping over the YAML config. Keep the
// values in sync with the YAML.
struct DrampoolMetricDef {
    const char* name;
    const char* type;
    const double* buckets{nullptr};
    std::size_t bucketCount{0};
};

inline const std::vector<DrampoolMetricDef>& DrampoolMetricDefs()
{
    static const std::vector<DrampoolMetricDef> kDefs = {
        // A. Request volume
        {kDumpRequestsTotal, "counter"},
        {kLoadRequestsTotal, "counter"},
        {kLookupRequestsTotal, "counter"},
        // B. Dump business
        {kDumpNospaceFailuresTotal, "counter"},
        {kDumpFailedEntriesTotal, "counter"},
        {kDumpPrepareDurationMs, "histogram", kMsBucketsPrepare, std::size(kMsBucketsPrepare)},
        // C. Load business
        {kLoadMissEntriesTotal, "counter"},
        {kLoadPrepareDurationMs, "histogram", kMsBucketsPrepare, std::size(kMsBucketsPrepare)},
        // D. Lookup business
        {kLookupMissEntriesTotal, "counter"},
        {kLookupScanDurationMs, "histogram", kMsBucketsScan, std::size(kMsBucketsScan)},
        // E. Transfer and response
        {kDumpTransferDurationMs, "histogram", kMsBucketsTransfer, std::size(kMsBucketsTransfer)},
        {kLoadTransferDurationMs, "histogram", kMsBucketsTransfer, std::size(kMsBucketsTransfer)},
        {kTransferFailuresTotal, "counter"},
        {kResponseRttMs, "histogram", kMsBucketsTransfer, std::size(kMsBucketsTransfer)},
        {kResponseFailuresTotal, "counter"},
        {kSubmitFailuresTotal, "counter"},
        // F. Resource usage
        {kMetadataEntryCount, "gauge"},
        {kFlagPoolUsageRatio, "gauge"},
        // G. Metadata settlement duration
        {kMetadataStoreendDurationMs, "histogram", kMsBucketsSettlement,
         std::size(kMsBucketsSettlement)},
        {kMetadataLoadendDurationMs, "histogram", kMsBucketsSettlement,
         std::size(kMsBucketsSettlement)},
        {kMetadataEvictGcDurationMs, "histogram", kMsBucketsGc, std::size(kMsBucketsGc)},
        // H. Queues and blocking
        {kQueueRequestFullTotal, "counter"},
        {kQueueRequestEnqueueWaitMs, "histogram", kMsBucketsSettlement,
         std::size(kMsBucketsSettlement)},
        {kQueueRequestSize, "gauge"},
        {kQueueCompletionFullTotal, "counter"},
        {kQueueCompletionInflight, "gauge"},
        {kQueueCompletionSize, "gauge"},
        {kQueueResponseBufferRetryTotal, "counter"},
        // I. Batch total duration
        {kDumpBatchTotalDurationMs, "histogram", kMsBucketsTransfer,
         std::size(kMsBucketsTransfer)},
        {kLoadBatchTotalDurationMs, "histogram", kMsBucketsTransfer,
         std::size(kMsBucketsTransfer)},
        {kLookupBatchTotalDurationMs, "histogram", kMsBucketsLookupBatch,
         std::size(kMsBucketsLookupBatch)},
    };
    return kDefs;
}

// One-shot metrics registration for the DramPool daemon. The daemon is a pure
// C++ process without the Python binding, so it creates the same names, types,
// and buckets as the UCM Python side (setup_ucm_metrics -> ucmmetrics) does
// from examples/metrics/metrics_configs.yaml: like the Python loop over the
// config, every entry of DrampoolMetricDefs() is passed to CreateStats, plus
// the dynamic per-slot-size gauges below. CreateStats is idempotent and first
// registration wins; unregistered names are silently dropped by UpdateStats
// (C2). Call after the runtime config is parsed so the dynamic per-slot-size
// gauges follow g_config.poolBlockSizes.
inline void SetupDrampoolMetrics()
{
    UC::Metrics::SetUp();
    for (const auto& def : DrampoolMetricDefs()) {
        if (def.buckets != nullptr) {
            UC::Metrics::CreateStats(def.name, def.type,
                                     {def.buckets, def.buckets + def.bucketCount});
        } else {
            UC::Metrics::CreateStats(def.name, def.type);
        }
    }
    // F-supplement: dynamic data-pool usage gauges, one per block size.
    for (const auto slotSize : g_config.poolBlockSizes) {
        UC::Metrics::CreateStats(BufferPoolUsageRatioName(slotSize), "gauge");
    }
}

}  // namespace UC::DramPool
