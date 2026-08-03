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
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>
#include "drampool_config.h"

namespace UC::DramPool {
namespace {

class RuntimeYamlFile {
public:
    explicit RuntimeYamlFile(const std::string& contents)
        : path_(std::filesystem::temp_directory_path() /
                ("drampool_runtime_" + std::to_string(sequence_++) + ".yaml"))
    {
        std::ofstream output(path_);
        output << contents;
    }

    ~RuntimeYamlFile()
    {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    const std::filesystem::path& Path() const { return path_; }

private:
    inline static std::uint64_t sequence_{0};
    std::filesystem::path path_;
};

std::string ValidRuntimeYaml()
{
    return R"(transport:
  device_ids:
    - 0
    - 2
  endpoints:
    - two_sided: "127.0.0.1:9000"
      one_sided: "127.0.0.1:4501"
    - two_sided: "127.0.0.1:9001"
      one_sided: "127.0.0.1:4502"
queue:
  request_depth: 65536
  completion_depth: 65536
request_receiver:
  idle_wait_us: 100
poller:
  pending_depth: 64
flag_buffer:
  capacity_mb: 64
  slot_size_bytes: 64
gc:
  enabled: true
  interval_ms: 1000
metadata:
  periodic_eviction_policy: TTL
  deep_eviction_policy: POSITION
  lease_time_ms: 5000
  default_evict_ratio: 0.0
  evict_period_ms: 31536000000
operation:
  timeout_ms: 5000
logger:
  level: info
  dir: ./logs
  max_files: 10
  max_size_mb: 5
)";
}

std::string ReplaceOnce(std::string text, const std::string& from, const std::string& to)
{
    const auto position = text.find(from);
    EXPECT_NE(position, std::string::npos);
    if (position != std::string::npos) { text.replace(position, from.size(), to); }
    return text;
}

DramPoolConfig LaunchConfig()
{
    DramPoolConfig config;
    config.runtimeConfigPath = "launch-selected.yaml";
    config.addr.host = "127.0.0.1";
    config.addr.port = 9000;
    config.poolSizeGb = 7;
    config.nics = {"mlx5_0"};
    return config;
}

struct InvalidYamlCase {
    const char* name;
    const char* original;
    const char* replacement;
    const char* expected;
};

class InvalidRuntimeYamlTest : public testing::TestWithParam<InvalidYamlCase> {};

TEST(DramPoolRuntimeYamlTest, LoadsEveryRuntimeFieldAndPreservesLaunchFields)
{
    auto config = LaunchConfig();
    RuntimeYamlFile yaml(ValidRuntimeYaml());

    const auto status = ParseYamlConfig(yaml.Path().string(), config);

    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(config.runtimeConfigPath, "launch-selected.yaml");
    EXPECT_EQ(config.addr.port, 9000U);
    EXPECT_EQ(config.poolSizeGb, 7U);
    EXPECT_EQ(config.nics, (std::vector<std::string>{"mlx5_0"}));
    ASSERT_EQ(config.twoSidedToOneSided.size(), 2U);
    EXPECT_EQ(config.twoSidedToOneSided.at("127.0.0.1:9000"), "127.0.0.1:4501");
    EXPECT_EQ(config.twoSidedToOneSided.at("127.0.0.1:9001"), "127.0.0.1:4502");
    EXPECT_EQ(config.transportDeviceIds, (std::vector<std::int32_t>{0, 2}));
    EXPECT_EQ(config.requestQueueDepth, 65536U);
    EXPECT_EQ(config.completionQueueDepth, 65536U);
    EXPECT_EQ(config.requestReceiverIdleWaitUs, 100U);
    EXPECT_EQ(config.pollerPendingDepth, 64U);
    EXPECT_EQ(config.flagBufferCapacityMb, 64U);
    EXPECT_EQ(config.flagBufferSlotSizeBytes, 64U);
    EXPECT_EQ(config.flagBufferSlotCount, 1'048'576U);
    EXPECT_TRUE(config.gcEnabled);
    EXPECT_EQ(config.gcIntervalMs, 1000U);
    EXPECT_EQ(config.metadataPeriodicEvictionPolicy, EvictionPolicyType::TTL);
    EXPECT_EQ(config.metadataDeepEvictionPolicy, EvictionPolicyType::POSITION);
    EXPECT_EQ(config.metadataLeaseTimeMs, 5000U);
    EXPECT_DOUBLE_EQ(config.metadataDefaultEvictRatio, 0.0);
    EXPECT_EQ(config.metadataEvictPeriodMs, 31'536'000'000ULL);
    EXPECT_EQ(config.opTimeoutMs, 5000U);
    EXPECT_EQ(config.logLevel, "info");
    EXPECT_EQ(config.logDir, "./logs");
    EXPECT_EQ(config.logMaxFiles, 10U);
    EXPECT_EQ(config.logMaxSizeMb, 5U);
}

TEST(DramPoolRuntimeYamlTest, LoadsConfiguredEvictionPoliciesCaseInsensitively)
{
    auto yamlText = ReplaceOnce(ValidRuntimeYaml(), "periodic_eviction_policy: TTL",
                                "periodic_eviction_policy: position");
    yamlText = ReplaceOnce(std::move(yamlText), "deep_eviction_policy: POSITION",
                           "deep_eviction_policy: ttl");
    auto config = LaunchConfig();
    RuntimeYamlFile yaml(yamlText);

    const auto status = ParseYamlConfig(yaml.Path().string(), config);

    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(config.metadataPeriodicEvictionPolicy, EvictionPolicyType::POSITION);
    EXPECT_EQ(config.metadataDeepEvictionPolicy, EvictionPolicyType::TTL);
}

TEST(DramPoolRuntimeYamlTest, RejectsMissingUnknownAndDuplicateKeys)
{
    const std::vector<std::pair<std::string, std::string>> cases = {
        {ReplaceOnce(ValidRuntimeYaml(), "  max_size_mb: 5\n", ""), "missing required key"},
        {ValidRuntimeYaml() + "unknown: 1\n", "unknown DramPool runtime YAML key"},
        {ValidRuntimeYaml() + "transport:\n  device_ids:\n    - 1\n", "duplicate YAML key"},
    };
    for (const auto& item : cases) {
        auto config = LaunchConfig();
        RuntimeYamlFile yaml(item.first);
        const auto status = ParseYamlConfig(yaml.Path().string(), config);
        EXPECT_TRUE(status.Failure());
        EXPECT_NE(status.ToString().find(item.second), std::string::npos) << status.ToString();
    }
}

TEST(DramPoolRuntimeYamlTest, FailureDoesNotPartiallyModifyConfiguration)
{
    auto config = LaunchConfig();
    config.transportDeviceIds = {37, 38};
    RuntimeYamlFile yaml(ReplaceOnce(ValidRuntimeYaml(), "  max_size_mb: 5", "  max_size_mb: 0"));

    EXPECT_TRUE(ParseYamlConfig(yaml.Path().string(), config).Failure());

    EXPECT_EQ(config.transportDeviceIds, (std::vector<std::int32_t>{37, 38}));
    EXPECT_TRUE(config.twoSidedToOneSided.empty());
    EXPECT_EQ(config.poolSizeGb, 7U);
}

TEST_P(InvalidRuntimeYamlTest, RejectsInvalidValue)
{
    const auto& item = GetParam();
    auto config = LaunchConfig();
    RuntimeYamlFile yaml(ReplaceOnce(ValidRuntimeYaml(), item.original, item.replacement));

    const auto status = ParseYamlConfig(yaml.Path().string(), config);

    EXPECT_TRUE(status.Failure());
    EXPECT_NE(status.ToString().find(item.expected), std::string::npos) << status.ToString();
}

INSTANTIATE_TEST_SUITE_P(
    Validation, InvalidRuntimeYamlTest,
    testing::Values(
        InvalidYamlCase{"MissingLocalEndpoint", "two_sided: \"127.0.0.1:9000\"",
                        "two_sided: \"127.0.0.1:9010\"", "no two_sided entry for --addr"},
        InvalidYamlCase{"DuplicateTwoSided", "two_sided: \"127.0.0.1:9001\"",
                        "two_sided: \"127.0.0.1:9000\"", "duplicate transport two_sided"},
        InvalidYamlCase{"DuplicateOneSided", "one_sided: \"127.0.0.1:4502\"",
                        "one_sided: \"127.0.0.1:4501\"", "duplicate transport one_sided"},
        InvalidYamlCase{"EndpointInBothRoles", "one_sided: \"127.0.0.1:4502\"",
                        "one_sided: \"127.0.0.1:9000\"", "both two_sided and one_sided"},
        InvalidYamlCase{"EmptyDeviceIds", "  device_ids:\n    - 0\n    - 2\n", "  device_ids:\n",
                        "must not be empty"},
        InvalidYamlCase{"NegativeDevice", "    - 0", "    - -1", "negative values"},
        InvalidYamlCase{"DuplicateDevice", "    - 2", "    - 0", "duplicate device ID"},
        InvalidYamlCase{"QueueTooShallow", "request_depth: 65536", "request_depth: 1",
                        "at least 2"},
        InvalidYamlCase{"ZeroPollerPendingDepth", "pending_depth: 64", "pending_depth: 0",
                        "greater than zero"},
        InvalidYamlCase{"PollerDepthExceedsFlagBufferSlots", "pending_depth: 64",
                        "pending_depth: 2000000", "at least poller.pending_depth slots"},
        InvalidYamlCase{"ZeroFlagBufferCapacity", "capacity_mb: 64", "capacity_mb: 0",
                        "must be greater than zero"},
        InvalidYamlCase{"ZeroFlagBufferSlotSize", "slot_size_bytes: 64", "slot_size_bytes: 0",
                        "must be greater than zero"},
        InvalidYamlCase{"EnabledGcHasZeroInterval", "interval_ms: 1000", "interval_ms: 0",
                        "when GC is enabled"},
        InvalidYamlCase{"UnsupportedPeriodicEvictionPolicy", "periodic_eviction_policy: TTL",
                        "periodic_eviction_policy: LRU", "unsupported eviction policy"},
        InvalidYamlCase{"UnsupportedDeepEvictionPolicy", "deep_eviction_policy: POSITION",
                        "deep_eviction_policy: LFU", "unsupported eviction policy"},
        InvalidYamlCase{"EvictRatioAboveOne", "default_evict_ratio: 0.0",
                        "default_evict_ratio: 1.1", "must be in [0, 1]"},
        InvalidYamlCase{"UnsupportedLogLevel", "level: info", "level: verbose", "unsupported"},
        InvalidYamlCase{"EmptyLogDirectory", "dir: ./logs", "dir: ''", "must not be empty"}),
    [](const testing::TestParamInfo<InvalidYamlCase>& info) { return info.param.name; });

}  // namespace
}  // namespace UC::DramPool
