#include "kv_metrics/ucm_metrics_adapter.h"
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
};

void ReportUpdateFailure(const char* metric, const char* reason)
{
    UC_ERROR("KV UCM metric update failed for {}; sample skipped: {}", metric, reason);
}

bool InstallationFailed(std::string* error, const char* reason) noexcept
{
    UC_ERROR(
        "Failed to install ASU UCM metrics backend: {}. "
        "Continuing without UCM metrics installation; existing backend unchanged. "
        "A later Store Setup can retry.",
        reason);
    // Even diagnostic string allocation must not make Store initialization fail.
    try {
        if (error != nullptr) { *error = reason; }
    } catch (...) {
    }
    return false;
}

class UcmMetricsBackend final : public KvMetricsBackend {
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
            UC::Metrics::UpdateStats(binding->nativeMetric, value);
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

std::shared_ptr<KvMetricsBackend> CreateUcmMetricsBackend()
{
    return std::make_shared<UcmMetricsBackend>();
}

bool EnsureUcmKvMetricsInstalled(std::string* error) noexcept
{
    try {
        static std::mutex mutex;
        // Retain the same backend across quiescent shutdown/reinstallation.
        // Its cached bindings remain valid across Store lifetimes.
        static std::shared_ptr<KvMetricsBackend> backend;
        std::lock_guard<std::mutex> lock{mutex};
        if (error != nullptr) { error->clear(); }
        if (!backend) { backend = CreateUcmMetricsBackend(); }
        UC_INFO("Initializing KV metrics backend: UcmMetricsBackend");
        std::string installError;
        if (!InstallBackend(backend, &installError)) {
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
