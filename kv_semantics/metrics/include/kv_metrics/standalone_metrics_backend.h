#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "kv_metrics/metrics.h"

namespace kv::metrics {

enum class MetricType { COUNTER = 0, GAUGE, HISTOGRAM };

struct MetricDescriptor {
    std::string name;
    MetricType type{MetricType::COUNTER};
    std::string documentation;
    std::vector<double> buckets;
    MetricLabels labels;
};

struct StandaloneMetricsConfig {
    std::string definitionPath;
    std::string metricPrefix{"kv:"};
    std::string listenAddress{"127.0.0.1"};
    std::uint16_t port{9108};
    std::string metricsPath{"/metrics"};
    std::uint32_t aggregationIntervalMs{500};
    std::map<std::string, std::string> constantLabels;
};

bool SetUpStandaloneMetrics(StandaloneMetricsConfig config, std::string* error = nullptr);

}  // namespace kv::metrics
