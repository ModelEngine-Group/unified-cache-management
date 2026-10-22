#include "kv_metrics/ucm_metrics_adapter.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include "logger.h"
#include "metrics_api.h"

namespace {
// Only this isolated test executable overrides allocation. Injection is local to
// the calling thread, so logger worker threads are unaffected.
thread_local unsigned allocationFailures = 0;
}  // namespace

void* operator new(std::size_t size)
{
    if (allocationFailures != 0) {
        --allocationFailures;
        throw std::bad_alloc{};
    }
    if (void* memory = std::malloc(size == 0 ? 1 : size)) { return memory; }
    throw std::bad_alloc{};
}

void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }

namespace kv::metrics {
namespace {

class ScopedMetricsDiagnostics {
public:
    ScopedMetricsDiagnostics()
    {
        const char* configured = std::getenv("UCM_LOG_RATE_LIMIT_ENABLE");
        hadSetting_ = configured != nullptr;
        previous_ = configured != nullptr ? configured : "true";
        setenv("UCM_LOG_RATE_LIMIT_ENABLE", "false", 1);
        UC::Logger::Logger::GetInstance().LoadRateLimitConfig();
        UC::Logger::Setup("log", 3, 5);
    }

    ~ScopedMetricsDiagnostics()
    {
        setenv("UCM_LOG_RATE_LIMIT_ENABLE", previous_.c_str(), 1);
        UC::Logger::Logger::GetInstance().LoadRateLimitConfig();
        if (!hadSetting_) { unsetenv("UCM_LOG_RATE_LIMIT_ENABLE"); }
    }

private:
    bool hadSetting_{false};
    std::string previous_;
};

std::string WaitForMetricDiagnostic(const std::string& message)
{
    UC::Logger::Flush();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    do {
        std::ifstream file{"log/ucm-" + std::to_string(getpid()) + ".log"};
        const std::string content{std::istreambuf_iterator<char>{file},
                                  std::istreambuf_iterator<char>{}};
        if (content.find(message) != std::string::npos) { return content; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
    return {};
}

class UcmKvMetricsInstallDeathTest : public testing::Test {
protected:
    void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

TEST_F(UcmKvMetricsInstallDeathTest, AllocationFailureDoesNotPoisonLaterInstallation)
{
    ASSERT_EXIT(
        {
            std::string error;
            allocationFailures = 1;
            const bool first = EnsureUcmKvMetricsInstalled(&error);
            allocationFailures = 0;
            EXPECT_FALSE(first);
            EXPECT_FALSE(IsEnabled());
            EXPECT_FALSE(error.empty());
            ASSERT_TRUE(EnsureUcmKvMetricsInstalled(&error));
            EXPECT_TRUE(error.empty());
            EXPECT_TRUE(IsEnabled());
            _exit(testing::Test::HasFailure() ? 1 : 0);
        },
        testing::ExitedWithCode(0), "");
}

TEST_F(UcmKvMetricsInstallDeathTest, DiagnosticAllocationFailureCannotEscapeInstallation)
{
    ASSERT_EXIT(
        {
            std::string error;
            allocationFailures = 8;
            const bool first = EnsureUcmKvMetricsInstalled(&error);
            allocationFailures = 0;
            EXPECT_FALSE(first);
            EXPECT_FALSE(IsEnabled());
            ASSERT_TRUE(EnsureUcmKvMetricsInstalled());
            _exit(testing::Test::HasFailure() ? 1 : 0);
        },
        testing::ExitedWithCode(0), "diagnostic logging also failed");
}

TEST_F(UcmKvMetricsInstallDeathTest, ConcurrentSetupSharesOneAdapter)
{
    ASSERT_EXIT(
        {
            std::atomic<unsigned> successes{0};
            std::vector<std::thread> callers;
            for (int i = 0; i < 8; ++i) {
                callers.emplace_back([&] {
                    if (EnsureUcmKvMetricsInstalled()) { ++successes; }
                });
            }
            for (auto& caller : callers) { caller.join(); }
            EXPECT_EQ(successes.load(), 8U);
            EXPECT_TRUE(IsEnabled());
            _exit(testing::Test::HasFailure() ? 1 : 0);
        },
        testing::ExitedWithCode(0), "");
}

TEST_F(UcmKvMetricsInstallDeathTest, DoesNotMistakeAnotherBackendForRetainedAdapter)
{
    ASSERT_EXIT(
        {
            ASSERT_TRUE(EnsureUcmKvMetricsInstalled());
            Shutdown();
            auto other = CreateUcmKvMetricsAdapter();
            ASSERT_TRUE(InstallBackend(other));
            std::string error;
            EXPECT_FALSE(EnsureUcmKvMetricsInstalled(&error));
            EXPECT_EQ(error, "metrics backend is already initialized");
            EXPECT_TRUE(IsBackendInstalled(other.get()));
            Shutdown();
            EXPECT_TRUE(EnsureUcmKvMetricsInstalled(&error));
            EXPECT_TRUE(error.empty());
            _exit(testing::Test::HasFailure() ? 1 : 0);
        },
        testing::ExitedWithCode(0), "");
}

TEST_F(UcmKvMetricsInstallDeathTest, DiagnosesUninitializedCollectorAndLaterRegistration)
{
    ASSERT_EXIT(
        {
            ScopedMetricsDiagnostics diagnostics;
            CachedMetric metric{"collector_startup_counter"};
            UC::Metrics::CachedMetric native{metric.Name()};
            EXPECT_EQ(UC::Metrics::TryUpdateStats(native, 100.0),
                      UC::Metrics::UpdateResult::NotInitialized);
            ASSERT_TRUE(InstallBackend(CreateUcmKvMetricsAdapter()));
            UpdateStats(metric, 100.0);
            const auto uninitialized =
                "KV UCM metric " + metric.Name() + " skipped: collector is not initialized";
            EXPECT_FALSE(WaitForMetricDiagnostic(uninitialized).empty());

            UC::Metrics::SetUp();
            EXPECT_EQ(UC::Metrics::TryUpdateStats(native, 100.0),
                      UC::Metrics::UpdateResult::Unregistered);
            UpdateStats(metric, 100.0);
            const auto unregistered =
                "KV UCM metric " + metric.Name() + " skipped: unregistered metric";
            EXPECT_FALSE(WaitForMetricDiagnostic(unregistered).empty());

            UC::Metrics::CreateStats(metric.Name(), "counter");
            EXPECT_EQ(UC::Metrics::TryUpdateStats(native, 2.0), UC::Metrics::UpdateResult::Updated);
            UpdateStats(metric, 3.0);
            EXPECT_DOUBLE_EQ(std::get<0>(UC::Metrics::GetAllStatsAndClear()).at(metric.Name()),
                             5.0);
            _exit(testing::Test::HasFailure() ? 1 : 0);
        },
        testing::ExitedWithCode(0), "");
}

class UcmKvMetricsAdapterTest : public testing::Test {
protected:
    void SetUp() override
    {
        UC::Metrics::SetUp();
        UC::Metrics::GetAllStatsAndClear();
        backend_ = CreateUcmKvMetricsAdapter();
        ASSERT_TRUE(InstallBackend(backend_));
        // UCM registrations persist in a process, including across --gtest_repeat.
        static unsigned sequence = 0;
        prefix_ = "adapter_test_" + std::to_string(++sequence) + "_";
    }

    void TearDown() override
    {
        Shutdown();
        UC::Metrics::GetAllStatsAndClear();
    }

    std::string Register(const std::string& suffix, const std::string& type,
                         const std::vector<double>& buckets = {})
    {
        const auto name = prefix_ + suffix;
        UC::Metrics::CreateStats(name, type, buckets);
        return name;
    }

    std::string prefix_;
    std::shared_ptr<KvMetricsBackend> backend_;
};

TEST_F(UcmKvMetricsAdapterTest, RoutesSingleUpdatesAndDrainsOnlyOnce)
{
    CachedMetric counter{Register("counter", "counter")};
    CachedMetric gauge{Register("gauge", "gauge")};
    CachedMetric histogram{Register("histogram", "histogram", {1.0, 5.0})};
    UpdateStats(counter, 2.0);
    UpdateStats(counter, 3.0);
    UpdateStats(gauge, 2.0);
    UpdateStats(gauge, 7.0);
    for (const auto value : {0.5, 3.0, 10.0}) { UpdateStats(histogram, value); }

    const auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_DOUBLE_EQ(std::get<0>(stats).at(counter.Name()), 5.0);
    EXPECT_DOUBLE_EQ(std::get<1>(stats).at(gauge.Name()), 7.0);
    const auto& buckets = std::get<2>(stats).at(histogram.Name());
    EXPECT_EQ(buckets.bucketCounts, (std::vector<uint64_t>{1, 1, 1}));
    EXPECT_DOUBLE_EQ(buckets.sum, 13.5);

    const auto drained = UC::Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(drained).count(counter.Name()), 0U);
    EXPECT_EQ(std::get<2>(drained).count(histogram.Name()), 0U);
}

TEST_F(UcmKvMetricsAdapterTest, RoutesBatchAndIgnoresNullEntries)
{
    CachedMetric counter{Register("counter", "counter")};
    CachedMetric gauge{Register("gauge", "gauge")};
    CachedMetric histogram{Register("histogram", "histogram", {1.0})};
    const MetricUpdate updates[] = {
        {counter,   2.0  },
        {nullptr,   100.0},
        {gauge,     7.0  },
        {histogram, 0.5  },
        {counter,   3.0  }
    };
    UpdateStats(updates, std::size(updates));
    UpdateStats(updates, 0);
    UpdateStats(nullptr, 5);
    // Exercise the backend's own guard as well as the facade guard.
    backend_->UpdateStats(nullptr, 5);
    const auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_DOUBLE_EQ(std::get<0>(stats).at(counter.Name()), 5.0);
    EXPECT_DOUBLE_EQ(std::get<1>(stats).at(gauge.Name()), 7.0);
    EXPECT_EQ(std::get<2>(stats).at(histogram.Name()).bucketCounts, (std::vector<uint64_t>{1, 0}));
    EXPECT_DOUBLE_EQ(std::get<2>(stats).at(histogram.Name()).sum, 0.5);
}

TEST_F(UcmKvMetricsAdapterTest, ResolvesSameCachedMetricAfterLateRegistration)
{
    CachedMetric metric{prefix_ + "late"};
    UpdateStats(metric, 100.0);
    EXPECT_EQ(std::get<0>(UC::Metrics::GetAllStatsAndClear()).count(metric.Name()), 0U);
    UC::Metrics::CreateStats(metric.Name(), "counter");
    UpdateStats(metric, 2.0);
    EXPECT_DOUBLE_EQ(std::get<0>(UC::Metrics::GetAllStatsAndClear()).at(metric.Name()), 2.0);
}

TEST_F(UcmKvMetricsAdapterTest, BoundsUnregisteredDiagnosticsAndRecovers)
{
    // Disable the logger's own limiter to exercise the adapter's pre-format gate.
    ScopedMetricsDiagnostics diagnostics;
    CachedMetric metric{prefix_ + "missing_catalog_entry"};
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) {
        writers.emplace_back([&] {
            for (int sample = 0; sample < 500; ++sample) { UpdateStats(metric, 100.0); }
        });
    }
    for (auto& writer : writers) { writer.join(); }
    const auto message = "KV UCM metric " + metric.Name() + " skipped: unregistered metric";
    // Wait for a trailing message so the async logger has processed earlier samples.
    const auto marker = prefix_ + "missing_diagnostics_drained";
    UC_WARN_UNLIMITED("{}", marker);
    const auto content = WaitForMetricDiagnostic(marker);
    const auto first = content.find(message);
    ASSERT_NE(first, std::string::npos);
    EXPECT_EQ(content.find(message, first + message.size()), std::string::npos);
    EXPECT_EQ(std::get<0>(UC::Metrics::GetAllStatsAndClear()).count(metric.Name()), 0U);

    UC::Metrics::CreateStats(metric.Name(), "counter");
    UpdateStats(metric, 2.0);
    EXPECT_DOUBLE_EQ(std::get<0>(UC::Metrics::GetAllStatsAndClear()).at(metric.Name()), 2.0);
}

TEST_F(UcmKvMetricsAdapterTest, AggregatesNodeHistogramsIntoBaseName)
{
    const auto name = Register("nodes", "histogram", {1.0, 5.0});
    CachedMetric first{name, {{"node_id", "1"}}};
    CachedMetric second{name, {{"node_id", "2"}}};
    EXPECT_FALSE(RegisterMetricLabels(name, first.Labels()));
    EXPECT_FALSE(RegisterMetricLabels(name, second.Labels()));
    UpdateStats(first, 0.5);
    UpdateStats(first, 3.0);
    UpdateStats(second, 4.0);
    UpdateStats(second, 10.0);
    const auto histograms = std::get<2>(UC::Metrics::GetAllStatsAndClear());
    ASSERT_EQ(histograms.size(), 1U);
    ASSERT_EQ(histograms.count(name), 1U);
    EXPECT_EQ(histograms.at(name).bucketCounts, (std::vector<uint64_t>{1, 2, 1}));
    EXPECT_DOUBLE_EQ(histograms.at(name).sum, 17.5);
}

TEST_F(UcmKvMetricsAdapterTest, FlushAndStopLeaveDrainingToUcm)
{
    CachedMetric metric{Register("counter", "counter")};
    UpdateStats(metric, 2.0);
    Flush();
    backend_->Stop();
    UpdateStats(metric, 3.0);
    Shutdown();
    EXPECT_FALSE(IsEnabled());
    EXPECT_DOUBLE_EQ(std::get<0>(UC::Metrics::GetAllStatsAndClear()).at(metric.Name()), 5.0);
}

TEST_F(UcmKvMetricsAdapterTest, DrainsUpdatesFromRetiredThreads)
{
    CachedMetric counter{Register("counter", "counter")};
    CachedMetric histogram{Register("histogram", "histogram", {1.0})};
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) {
        writers.emplace_back([&] {
            for (int update = 0; update < 250; ++update) {
                const MetricUpdate updates[] = {
                    {counter,   1.0},
                    {histogram, 0.5}
                };
                UpdateStats(updates, std::size(updates));
            }
        });
    }
    for (auto& writer : writers) { writer.join(); }
    const auto stats = UC::Metrics::GetAllStatsAndClear();
    EXPECT_DOUBLE_EQ(std::get<0>(stats).at(counter.Name()), 1000.0);
    EXPECT_EQ(std::get<2>(stats).at(histogram.Name()).bucketCounts,
              (std::vector<uint64_t>{1000, 0}));
    EXPECT_DOUBLE_EQ(std::get<2>(stats).at(histogram.Name()).sum, 500.0);
}

TEST_F(UcmKvMetricsAdapterTest, LogsRepeatedUpdateFailuresAndRecovers)
{
    // Disable rate limiting only in this test to distinguish recurring logger
    // diagnostics from the old process-wide, one-shot stderr message.
    const char* configured = std::getenv("UCM_LOG_RATE_LIMIT_ENABLE");
    const bool hadSetting = configured != nullptr;
    const std::string previous = configured != nullptr ? configured : "true";
    setenv("UCM_LOG_RATE_LIMIT_ENABLE", "false", 1);
    UC::Logger::Logger::GetInstance().LoadRateLimitConfig();
    UC::Logger::Setup("log", 3, 5);
    CachedMetric metric{Register("failed_update", "counter")};
    for (int i = 0; i < 2; ++i) {
        allocationFailures = 1;
        UpdateStats(metric, 100.0);
        allocationFailures = 0;
    }
    UpdateStats(metric, 3.0);
    EXPECT_DOUBLE_EQ(std::get<0>(UC::Metrics::GetAllStatsAndClear()).at(metric.Name()), 3.0);
    UC::Logger::Flush();
    const std::string message = "KV UCM metric update failed for " + metric.Name();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    bool reportedTwice = false;
    do {
        std::ifstream file{"log/ucm-" + std::to_string(getpid()) + ".log"};
        const std::string content{std::istreambuf_iterator<char>{file},
                                  std::istreambuf_iterator<char>{}};
        const auto first = content.find(message);
        reportedTwice = first != std::string::npos &&
                        content.find(message, first + message.size()) != std::string::npos;
        if (reportedTwice) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
    setenv("UCM_LOG_RATE_LIMIT_ENABLE", previous.c_str(), 1);
    UC::Logger::Logger::GetInstance().LoadRateLimitConfig();
    if (!hadSetting) { unsetenv("UCM_LOG_RATE_LIMIT_ENABLE"); }
    EXPECT_TRUE(reportedTwice);
}

TEST_F(UcmKvMetricsAdapterTest, LoggingAllocationFailureDoesNotEscapeUpdate)
{
    CachedMetric metric{Register("logging_failure", "counter")};
    allocationFailures = 2;
    UpdateStats(metric, 100.0);
    allocationFailures = 0;
    UpdateStats(metric, 4.0);
    EXPECT_DOUBLE_EQ(std::get<0>(UC::Metrics::GetAllStatsAndClear()).at(metric.Name()), 4.0);
}

TEST_F(UcmKvMetricsAdapterTest, SkipsForeignCachedBindingWithoutInvalidCast)
{
    struct ForeignBinding : CachedMetric::Binding {
        explicit ForeignBinding(Kind kind) : CachedMetric::Binding(kind) {}
    };
    for (const auto kind :
         {CachedMetric::Binding::Kind::Unknown, CachedMetric::Binding::Kind::Standalone}) {
        CachedMetric metric{
            Register("foreign_binding_" + std::to_string(static_cast<int>(kind)), "counter")};
        metric.Resolve([kind] { return std::make_unique<ForeignBinding>(kind); });
        UpdateStats(metric, 100.0);
        EXPECT_EQ(std::get<0>(UC::Metrics::GetAllStatsAndClear()).count(metric.Name()), 0U);
    }
}

}  // namespace
}  // namespace kv::metrics
