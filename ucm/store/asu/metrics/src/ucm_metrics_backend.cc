#include "kv_metrics/ucm_metrics_backend.h"
#include <exception>
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

class UcmMetricsBackend final : public KvMetricsBackend {
public:
    void UpdateStats(CachedMetric& metric, double value) noexcept override
    {
        try {
            auto* cached = metric.Resolve(
                [&metric] { return std::make_unique<UcmMetricBinding>(metric.Name()); });
            if (cached == nullptr || cached->GetKind() != CachedMetric::Binding::Kind::Ucm) {
                UC_ERROR("KV UCM metric update failed for {}; sample skipped: {}", metric.Name(),
                         "incompatible cached metric binding");
                return;
            }
            auto* binding = static_cast<UcmMetricBinding*>(cached);
            // UCM resolves late registration using its own cached ID and epoch.
            // Node labels deliberately aggregate into the base name in this backend.
            UC::Metrics::UpdateStats(binding->nativeMetric, value);
        } catch (const std::exception& ex) {
            UC_ERROR("KV UCM metric update failed for {}; sample skipped: {}", metric.Name(),
                     ex.what());
        } catch (...) {
            UC_ERROR("KV UCM metric update failed for {}; sample skipped: {}", metric.Name(),
                     "unknown exception");
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

void EnsureUcmKvMetricsInstalled() noexcept
{
    try {
        // Thread-safe initialization retries after an exception. Retain the same
        // backend across quiescent shutdown/reinstallation and Store lifetimes.
        static const auto backend = CreateUcmMetricsBackend();
        std::string installError;
        if (!InstallBackend(backend, &installError)) {
            if (IsEnabled()) {
                UC_INFO("KV metrics backend already initialized; retaining the existing backend");
                return;
            }
            UC_ERROR(
                "Failed to install ASU UCM metrics backend: {}. Store setup continues; "
                "existing backend unchanged; a later Store Setup can retry.",
                installError);
            return;
        }
        UC_INFO("Installed KV metrics backend: UcmMetricsBackend");
    } catch (const std::exception& ex) {
        UC_ERROR(
            "Failed to install ASU UCM metrics backend: {}. Store setup continues; "
            "a later Store Setup can retry.",
            ex.what());
    } catch (...) {
        UC_ERROR(
            "Failed to install ASU UCM metrics backend: unknown exception. "
            "Store setup continues; a later Store Setup can retry.");
    }
}

}  // namespace kv::metrics
