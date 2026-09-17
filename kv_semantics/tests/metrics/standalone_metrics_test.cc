#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include "kv_metrics/metrics.h"
#include "kv_metrics/standalone_metrics_backend.h"

namespace kv::metrics {
namespace {

std::uint16_t FindUnusedLoopbackPort()
{
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    EXPECT_GE(fd, 0);
    if (fd < 0) { return 0; }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    EXPECT_EQ(bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)), 0);

    socklen_t length = sizeof(address);
    EXPECT_EQ(getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length), 0);
    const auto port = ntohs(address.sin_port);
    close(fd);
    return port;
}

std::string HttpGet(std::uint16_t port, const std::string& path)
{
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    EXPECT_GE(fd, 0);
    if (fd < 0) { return {}; }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    EXPECT_EQ(connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)), 0);

    const auto request = "GET " + path + " HTTP/1.1\r\nHost: localhost\r\n\r\n";
    EXPECT_EQ(send(fd, request.data(), request.size(), 0), static_cast<ssize_t>(request.size()));

    std::string response;
    char buffer[4096];
    while (true) {
        const auto count = recv(fd, buffer, sizeof(buffer), 0);
        if (count <= 0) { break; }
        response.append(buffer, static_cast<std::size_t>(count));
    }
    close(fd);
    return response;
}

std::filesystem::path WriteTestMetricConfig(std::uint16_t port, const std::string& prefix = "kv:")
{
    const auto path = std::filesystem::temp_directory_path() /
                      ("kv-metrics-inline-" + std::to_string(port) + ".yaml");
    std::ofstream output{path};
    EXPECT_TRUE(output.is_open());
    output << "metric_prefix: \"" << prefix << "\"\n"
           << "counter:\n"
           << "  - {name: \"test_counter\", documentation: \"Test counter\"}\n"
           << "gauge:\n"
           << "  - {name: \"test_gauge\", documentation: \"Test gauge\"}\n"
           << "histogram:\n"
           << "  - {name: \"test_histogram\", documentation: \"Test histogram\", "
              "buckets: [0.001, 0.01, 0.1]}\n";
    return path;
}

class StandaloneMetricsTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        port_ = FindUnusedLoopbackPort();
        ASSERT_NE(port_, 0);
        configPath_ = WriteTestMetricConfig(port_);
    }

    void TearDown() override
    {
        Shutdown();
        std::error_code error;
        std::filesystem::remove(configPath_, error);
    }

    StandaloneMetricsConfig MakeConfig() const
    {
        StandaloneMetricsConfig config;
        config.port = port_;
        config.definitionPath = configPath_.string();
        return config;
    }

    std::uint16_t port_{0};
    std::filesystem::path configPath_;
};

class RecordingKvMetricsBackend final : public KvMetricsBackend {
public:
    void UpdateStats(CachedMetric& metric, double value) noexcept override
    {
        const MetricUpdate update{metric, value};
        UpdateStats(&update, 1);
    }
    void UpdateStats(const MetricUpdate* updates, std::size_t count) noexcept override
    {
        for (std::size_t index = 0; index < count && recordedCount < recordedUpdates.size();
             ++index) {
            recordedUpdates[recordedCount++] = updates[index];
        }
    }
    std::array<MetricUpdate, 4> recordedUpdates{};
    std::size_t recordedCount{0};
};

TEST(MetricsFacadeTest, InstallsOneBackendAndDispatchesUpdates)
{
    auto backend = std::make_shared<RecordingKvMetricsBackend>();
    std::string error;
    ASSERT_TRUE(InstallBackend(backend, &error)) << error;
    UpdateStats(KV_METRIC("test_counter"), 1.0);
    ASSERT_EQ(backend->recordedCount, 1U);
    EXPECT_EQ(backend->recordedUpdates[0].metric->Name(), "test_counter");
    EXPECT_DOUBLE_EQ(backend->recordedUpdates[0].value, 1.0);
    Shutdown();
}

TEST_F(StandaloneMetricsTest, ExposesCounterGaugeAndHistogramInPrometheusFormat)
{
    auto config = MakeConfig();
    config.constantLabels = {
        {"source", "test"}
    };

    std::string error;
    ASSERT_TRUE(SetUpStandaloneMetrics(config, &error)) << error;

    UpdateStats(KV_METRIC("test_counter"), 2.0);
    UpdateStats(KV_METRIC("test_histogram"), 0.002);
    UpdateStats(KV_METRIC("test_gauge"), 2.0);
    Flush();

    const auto response = HttpGet(config.port, config.metricsPath);
    EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
    EXPECT_NE(response.find("kv:test_counter{source=\"test\"} 2"), std::string::npos);
    EXPECT_NE(response.find("kv:test_histogram_count{source=\"test\"} 1"), std::string::npos);
    EXPECT_NE(response.find("kv:test_gauge{source=\"test\"} 2"), std::string::npos);

    Shutdown();
}

TEST_F(StandaloneMetricsTest, AggregatesMetricsWrittenByMultipleThreads)
{
    auto config = MakeConfig();

    std::string error;
    ASSERT_TRUE(SetUpStandaloneMetrics(config, &error)) << error;

    constexpr int kThreadCount = 4;
    constexpr int kUpdatesPerThread = 1000;
    std::vector<std::thread> writers;
    writers.reserve(kThreadCount);
    for (int thread = 0; thread < kThreadCount; ++thread) {
        writers.emplace_back([] {
            for (int update = 0; update < kUpdatesPerThread; ++update) {
                UpdateStats(KV_METRIC("test_counter"), 1.0);
                UpdateStats(KV_METRIC("test_histogram"), 0.002);
            }
        });
    }
    for (auto& writer : writers) { writer.join(); }
    Flush();

    const auto response = HttpGet(config.port, config.metricsPath);
    EXPECT_NE(response.find("kv:test_counter 4000"), std::string::npos);
    EXPECT_NE(response.find("kv:test_histogram_count 4000"), std::string::npos);

    Shutdown();
}

TEST_F(StandaloneMetricsTest, DrainsRetiredThreadAfterPreviousSlotWasAggregated)
{
    auto config = MakeConfig();

    std::string error;
    ASSERT_TRUE(SetUpStandaloneMetrics(config, &error)) << error;

    std::atomic<int> phase{0};
    std::thread writer{[&phase] {
        UpdateStats(KV_METRIC("test_counter"), 1.0);
        phase.store(1, std::memory_order_release);
        while (phase.load(std::memory_order_acquire) != 2) { std::this_thread::yield(); }
        UpdateStats(KV_METRIC("test_counter"), 2.0);
    }};

    while (phase.load(std::memory_order_acquire) != 1) { std::this_thread::yield(); }
    Flush();
    phase.store(2, std::memory_order_release);
    writer.join();
    Flush();

    const auto response = HttpGet(config.port, config.metricsPath);
    EXPECT_NE(response.find("kv:test_counter 3"), std::string::npos);

    Shutdown();
}

TEST_F(StandaloneMetricsTest, RegistersInlineYamlDefinitionsFromConfiguredSource)
{
    auto config = MakeConfig();
    configPath_ = WriteTestMetricConfig(config.port, "configured:");
    config.definitionPath = configPath_.string();

    std::string error;
    ASSERT_TRUE(SetUpStandaloneMetrics(config, &error)) << error;
    UpdateStats(KV_METRIC("test_counter"), 3.0);
    Flush();

    const auto response = HttpGet(config.port, config.metricsPath);
    EXPECT_NE(response.find("configured:test_counter 3"), std::string::npos);

    Shutdown();
}

TEST_F(StandaloneMetricsTest, DoesNotLoseUpdatesDuringConcurrentFlush)
{
    auto config = MakeConfig();
    config.aggregationIntervalMs = 1;

    std::string error;
    ASSERT_TRUE(SetUpStandaloneMetrics(config, &error)) << error;

    constexpr int kThreadCount = 4;
    constexpr int kUpdatesPerThread = 2000;
    constexpr int kExpectedUpdates = kThreadCount * kUpdatesPerThread;
    std::atomic<int> finished{0};
    std::vector<std::thread> writers;
    writers.reserve(kThreadCount);
    for (int thread = 0; thread < kThreadCount; ++thread) {
        writers.emplace_back([&finished] {
            const MetricUpdate updates[] = {
                {KV_METRIC("test_counter"),   1.0  },
                {KV_METRIC("test_histogram"), 0.001},
            };
            for (int update = 0; update < kUpdatesPerThread; ++update) {
                UpdateStats(updates, std::size(updates));
            }
            finished.fetch_add(1, std::memory_order_release);
        });
    }

    while (finished.load(std::memory_order_acquire) != kThreadCount) {
        Flush();
        std::this_thread::yield();
    }
    for (auto& writer : writers) { writer.join(); }
    Flush();

    const auto response = HttpGet(config.port, config.metricsPath);
    EXPECT_NE(response.find("kv:test_counter " + std::to_string(kExpectedUpdates)),
              std::string::npos);
    EXPECT_NE(response.find("kv:test_histogram_count " + std::to_string(kExpectedUpdates)),
              std::string::npos);

    Shutdown();
}

}  // namespace
}  // namespace kv::metrics
