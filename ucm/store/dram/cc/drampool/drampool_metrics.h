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
#include <cstddef>
#include <cstdint>
#include <string>
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

// Histogram observation capacity per thread (UC::Metrics C3 cap). Sized to
// hold at least two 10s reporter windows of batch-level observations.
inline constexpr std::size_t kMetricsMaxVectorLen = 10000;

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
inline std::string BufferPoolUsageRatioName(std::uint64_t slotSize)
{
    return std::string("drampool_buffer_pool_usage_ratio_") + std::to_string(slotSize);
}

// ---- update helpers (thin wrappers over UC::Metrics::UpdateStats) ----
// Accumulates a count (COUNTER semantics: value += count); zero is skipped.
inline void MetricsCount(const std::string& name, std::uint64_t count)
{
    if (count != 0) {
        UC::Metrics::UpdateStats(name, static_cast<double>(count));
    }
}

// Records one duration sample (HISTOGRAM semantics: push_back, in ms).
inline void MetricsObserve(const std::string& name, double valueMs)
{
    UC::Metrics::UpdateStats(name, valueMs);
}

// Overwrites the latest value (GAUGE semantics: value = v).
inline void MetricsSet(const std::string& name, double value)
{
    UC::Metrics::UpdateStats(name, value);
}

// RAII duration observer: measures with SteadyNowUs() and records the elapsed
// time in ms on scope exit. Call Disarm() on paths that must not be observed
// (e.g. failed preparations, see metrics_design.md §4.1).
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name) : name_(name), startUs_(SteadyNowUs()) {}

    ~ScopedTimer()
    {
        if (armed_) {
            MetricsObserve(name_, static_cast<double>(SteadyNowUs() - startUs_) / 1000.0);
        }
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    void Disarm() { armed_ = false; }

private:
    std::string name_;
    std::uint64_t startUs_;
    bool armed_{true};
};

// Registers every metric name once at startup (C2: unregistered names are
// silently dropped by UpdateStats). Call after the runtime config is parsed.
inline void SetupDrampoolMetrics()
{
    UC::Metrics::SetUp(kMetricsMaxVectorLen);
    // A. Request volume
    UC::Metrics::CreateStats(kDumpRequestsTotal, "counter");
    UC::Metrics::CreateStats(kLoadRequestsTotal, "counter");
    UC::Metrics::CreateStats(kLookupRequestsTotal, "counter");
    // B. Dump business
    UC::Metrics::CreateStats(kDumpNospaceFailuresTotal, "counter");
    UC::Metrics::CreateStats(kDumpFailedEntriesTotal, "counter");
    UC::Metrics::CreateStats(kDumpPrepareDurationMs, "histogram");
    // C. Load business
    UC::Metrics::CreateStats(kLoadMissEntriesTotal, "counter");
    UC::Metrics::CreateStats(kLoadPrepareDurationMs, "histogram");
    // D. Lookup business
    UC::Metrics::CreateStats(kLookupMissEntriesTotal, "counter");
    UC::Metrics::CreateStats(kLookupScanDurationMs, "histogram");
    // E. Transfer and response
    UC::Metrics::CreateStats(kDumpTransferDurationMs, "histogram");
    UC::Metrics::CreateStats(kLoadTransferDurationMs, "histogram");
    UC::Metrics::CreateStats(kTransferFailuresTotal, "counter");
    UC::Metrics::CreateStats(kResponseRttMs, "histogram");
    UC::Metrics::CreateStats(kResponseFailuresTotal, "counter");
    UC::Metrics::CreateStats(kSubmitFailuresTotal, "counter");
    // F. Resource usage (data pools are registered dynamically below)
    UC::Metrics::CreateStats(kMetadataEntryCount, "gauge");
    UC::Metrics::CreateStats(kFlagPoolUsageRatio, "gauge");
    // G. Metadata settlement duration
    UC::Metrics::CreateStats(kMetadataStoreendDurationMs, "histogram");
    UC::Metrics::CreateStats(kMetadataLoadendDurationMs, "histogram");
    UC::Metrics::CreateStats(kMetadataEvictGcDurationMs, "histogram");
    // H. Queues and blocking
    UC::Metrics::CreateStats(kQueueRequestFullTotal, "counter");
    UC::Metrics::CreateStats(kQueueRequestEnqueueWaitMs, "histogram");
    UC::Metrics::CreateStats(kQueueRequestSize, "gauge");
    UC::Metrics::CreateStats(kQueueCompletionFullTotal, "counter");
    UC::Metrics::CreateStats(kQueueCompletionInflight, "gauge");
    UC::Metrics::CreateStats(kQueueCompletionSize, "gauge");
    UC::Metrics::CreateStats(kQueueResponseBufferRetryTotal, "counter");
    // I. Batch total duration
    UC::Metrics::CreateStats(kDumpBatchTotalDurationMs, "histogram");
    UC::Metrics::CreateStats(kLoadBatchTotalDurationMs, "histogram");
    UC::Metrics::CreateStats(kLookupBatchTotalDurationMs, "histogram");
    // F. Dynamic data-pool usage gauges, one per block size.
    for (const auto slotSize : g_config.poolBlockSizes) {
        UC::Metrics::CreateStats(BufferPoolUsageRatioName(slotSize), "gauge");
    }
}

}  // namespace UC::DramPool
