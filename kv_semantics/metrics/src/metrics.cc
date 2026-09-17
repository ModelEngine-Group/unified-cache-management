#include "kv_metrics/metrics.h"
#include <atomic>
#include <mutex>
#include <utility>

namespace kv::metrics {
namespace {

std::mutex gBackendMutex;
std::shared_ptr<KvMetricsBackend> gBackend;

std::atomic<KvMetricsBackend*> gBackendFast{nullptr};

std::shared_ptr<KvMetricsBackend> LoadBackend()
{
    return std::atomic_load_explicit(&gBackend, std::memory_order_acquire);
}

KvMetricsBackend* LoadBackendFast() noexcept
{
    return gBackendFast.load(std::memory_order_acquire);
}

}  // namespace

bool InstallBackend(std::shared_ptr<KvMetricsBackend> backend, std::string* error)
{
    if (!backend) {
        if (error != nullptr) { *error = "metrics backend is null"; }
        return false;
    }

    std::lock_guard<std::mutex> lock{gBackendMutex};
    if (LoadBackend()) {
        if (error != nullptr) { *error = "metrics backend is already initialized"; }
        return false;
    }
    auto* const backendFast = backend.get();
    std::atomic_store_explicit(&gBackend, std::move(backend), std::memory_order_release);
    gBackendFast.store(backendFast, std::memory_order_release);
    return true;
}

void Shutdown()
{
    gBackendFast.store(nullptr, std::memory_order_release);
    std::shared_ptr<KvMetricsBackend> backend;
    {
        std::lock_guard<std::mutex> lock{gBackendMutex};
        backend = std::atomic_exchange_explicit(&gBackend, std::shared_ptr<KvMetricsBackend>{},
                                                std::memory_order_acq_rel);
    }
    if (backend) {
        backend->Flush();
        backend->Stop();
    }
}

void Flush()
{
    auto backend = LoadBackend();
    if (backend) { backend->Flush(); }
}

bool IsEnabled() noexcept { return LoadBackendFast() != nullptr; }

MetricTimer StartMetricTimer() noexcept
{
    if (!IsEnabled()) { return std::nullopt; }
    return std::chrono::steady_clock::now();
}

void UpdateStats(CachedMetric& metric, double value) noexcept
{
    auto* backend = LoadBackendFast();
    if (backend) { backend->UpdateStats(metric, value); }
}

void UpdateStats(const MetricUpdate* updates, std::size_t count) noexcept
{
    if (updates == nullptr || count == 0) { return; }
    auto* backend = LoadBackendFast();
    if (backend) { backend->UpdateStats(updates, count); }
}

}  // namespace kv::metrics
