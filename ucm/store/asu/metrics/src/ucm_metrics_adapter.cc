#include "kv_metrics/ucm_metrics_adapter.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <mutex>
#include "logger.h"
#include "metrics_api.h"

namespace kv::metrics {
namespace {

struct UcmMetricBinding final : CachedMetric::Binding {
    explicit UcmMetricBinding(const std::string& name)
        : CachedMetric::Binding(Kind::Ucm), nativeMetric{name}
    {
    }
    UC::Metrics::CachedMetric nativeMetric;
    std::atomic<int64_t> nextUninitializedReport{0};
    std::atomic<int64_t> nextUnregisteredReport{0};
};

void ReportLoggingFailure() noexcept
{
    // Allocation-free fallback if the normal logger also fails (e.g. bad_alloc).
    // Keep it periodic, but bounded even during sustained memory pressure.
    static std::atomic<int64_t> nextReport{0};
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    auto next = nextReport.load(std::memory_order_relaxed);
    if (now >= next &&
        nextReport.compare_exchange_strong(next, now + 60, std::memory_order_relaxed)) {
        std::fputs("KV UCM metrics failure; diagnostic logging also failed.\n", stderr);
    }
}

void ReportUpdateFailure(const char* metric, const char* reason) noexcept
{
    try {
        UC_ERROR("KV UCM metric update failed for {}; sample skipped: {}", metric, reason);
    } catch (...) {
        ReportLoggingFailure();
    }
}

void ReportSkippedUpdate(UcmMetricBinding& binding, UC::Metrics::UpdateResult result) noexcept
{
    // Only the skipped-update path reads the clock. Gate before formatting so a
    // missing catalog entry does not allocate a log message for every sample.
    auto& nextReport = result == UC::Metrics::UpdateResult::NotInitialized
                           ? binding.nextUninitializedReport
                           : binding.nextUnregisteredReport;
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    auto next = nextReport.load(std::memory_order_relaxed);
    if (now < next ||
        !nextReport.compare_exchange_strong(next, now + 60, std::memory_order_relaxed)) {
        return;
    }
    try {
        if (result == UC::Metrics::UpdateResult::NotInitialized) {
            UC_WARN(
                "KV UCM metric {} skipped: collector is not initialized; call UCM metrics "
                "SetUp and register the catalog before sampling.",
                binding.nativeMetric.name);
        } else {
            UC_WARN(
                "KV UCM metric {} skipped: unregistered metric; check that the effective "
                "metrics catalog includes the KV definitions. Later registration can "
                "restore sampling.",
                binding.nativeMetric.name);
        }
    } catch (...) {
        ReportLoggingFailure();
    }
}

bool InstallationFailed(std::string* error, const char* reason) noexcept
{
    try {
        UC_ERROR(
            "Failed to install ASU UCM metrics backend: {}. "
            "Continuing without UCM metrics installation; existing backend unchanged. "
            "A later Store Setup can retry.",
            reason);
    } catch (...) {
        ReportLoggingFailure();
    }
    // Even diagnostic string allocation must not make Store initialization fail.
    try {
        if (error != nullptr) { *error = reason; }
    } catch (...) {
    }
    return false;
}

class UcmKvMetricsAdapter final : public KvMetricsBackend {
public:
    void UpdateStats(CachedMetric& metric, double value) noexcept override
    {
        try {
            auto* cached = metric.Resolve(
                [&metric] { return std::make_unique<UcmMetricBinding>(metric.Name()); });
            if (cached == nullptr || cached->GetKind() != CachedMetric::Binding::Kind::Ucm) {
                ReportUpdateFailure(metric.Name().c_str(), "incompatible cached metric binding");
                return;
            }
            auto* binding = static_cast<UcmMetricBinding*>(cached);
            // UCM resolves late registration using its own cached ID and epoch.
            // Node labels deliberately aggregate into the base name in this backend.
            const auto result = UC::Metrics::TryUpdateStats(binding->nativeMetric, value);
            if (result != UC::Metrics::UpdateResult::Updated) {
                ReportSkippedUpdate(*binding, result);
            }
        } catch (const std::exception& ex) {
            ReportUpdateFailure(metric.Name().c_str(), ex.what());
        } catch (...) {
            ReportUpdateFailure(metric.Name().c_str(), "unknown exception");
        }
    }

    void UpdateStats(const MetricUpdate* updates, std::size_t count) noexcept override
    {
        if (updates == nullptr) { return; }
        for (std::size_t i = 0; i < count; ++i) {
            if (updates[i].metric != nullptr) { UpdateStats(*updates[i].metric, updates[i].value); }
        }
    }

    bool RegisterMetricLabels(const std::string&, const MetricLabels&) noexcept override
    {
        return false;
    }

    // Flush and Stop inherit no-ops. Only the Python dispatcher owns draining.
};

}  // namespace

std::shared_ptr<KvMetricsBackend> CreateUcmKvMetricsAdapter()
{
    return std::make_shared<UcmKvMetricsAdapter>();
}

bool EnsureUcmKvMetricsInstalled(std::string* error) noexcept
{
    try {
        static std::mutex mutex;
        // Retain the same adapter across quiescent shutdown/reinstallation.
        // Its cached bindings remain valid across Store lifetimes.
        static std::shared_ptr<KvMetricsBackend> adapter;
        std::lock_guard<std::mutex> lock{mutex};
        if (error != nullptr) { error->clear(); }
        if (!adapter) { adapter = CreateUcmKvMetricsAdapter(); }
        if (IsBackendInstalled(adapter.get())) { return true; }
        std::string installError;
        if (!InstallBackend(adapter, &installError)) {
            return InstallationFailed(error, installError.c_str());
        }
        return true;
    } catch (const std::exception& ex) {
        return InstallationFailed(error, ex.what());
    } catch (...) {
        return InstallationFailed(error, "unknown exception");
    }
}

}  // namespace kv::metrics
