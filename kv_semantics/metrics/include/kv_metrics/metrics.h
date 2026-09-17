#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace kv::metrics {

class CachedMetric {
public:
    explicit CachedMetric(std::string name) : name_(std::move(name)) {}
    const std::string& Name() const noexcept { return name_; }

    struct Binding {
        virtual ~Binding() = default;
    };

    template <typename Factory>
    Binding* Resolve(Factory&& factory)
    {
        auto* binding = binding_.load(std::memory_order_acquire);
        if (binding != nullptr) { return binding; }
        std::lock_guard<std::mutex> lock{mutex_};
        binding = binding_.load(std::memory_order_relaxed);
        if (binding != nullptr) { return binding; }
        auto replacement = factory();
        binding = replacement.get();
        owner_ = std::move(replacement);
        binding_.store(binding, std::memory_order_release);
        return binding;
    }

private:
    const std::string name_;
    std::mutex mutex_;
    std::unique_ptr<Binding> owner_;
    std::atomic<Binding*> binding_{nullptr};
};

struct MetricUpdate {
    CachedMetric* metric{nullptr};
    double value{0.0};
    MetricUpdate() = default;
    MetricUpdate(CachedMetric& metric, double value) : metric(&metric), value(value) {}
    MetricUpdate(CachedMetric* metric, double value) : metric(metric), value(value) {}
};

using MetricTimer = std::optional<std::chrono::steady_clock::time_point>;

class KvMetricsBackend {
public:
    virtual ~KvMetricsBackend() = default;
    virtual void UpdateStats(CachedMetric& metric, double value) noexcept = 0;
    virtual void UpdateStats(const MetricUpdate* updates, std::size_t count) noexcept = 0;
    virtual void Flush() {}
    virtual void Stop() {}
};

bool InstallBackend(std::shared_ptr<KvMetricsBackend> backend, std::string* error = nullptr);
void Shutdown();
void Flush();
bool IsEnabled() noexcept;
MetricTimer StartMetricTimer() noexcept;
inline std::optional<double> ElapsedSeconds(const MetricTimer& timer) noexcept
{
    if (!timer) { return std::nullopt; }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - *timer).count();
}

void UpdateStats(CachedMetric& metric, double value) noexcept;
void UpdateStats(const MetricUpdate* updates, std::size_t count) noexcept;

}  // namespace kv::metrics

#define KV_METRIC(name)                                  \
    []() -> ::kv::metrics::CachedMetric& {               \
        static ::kv::metrics::CachedMetric metric{name}; \
        return metric;                                   \
    }()
