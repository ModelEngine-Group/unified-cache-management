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

// TransConfigUtils / 结构化配置 DT.
//
// 测试因子分解：
//   F1 配置结构 ∈ {Retry, Recovery, Probe, FlowControl, Connection}
//   F2 取值类别 ∈ {设计默认值, 合法边界(0/1/16/1024/4096), 越界(17/1025/4097),
//                  非法零值, 非法文本, 非法关系约束(busyFloor>=maxInflight)}
//   F3 入口     ∈ {Validate() 直调, LoadTransportConfig 文件, 客户端 LoadFromFile 文件}
// 断言面：Validate 返回码（非法值 INVALID_ARGUMENT，禁止静默回退）、
//         默认值快照、文件解析字段落位、legacy 键同步、派生上限。

#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include "kv_types.h"
#include "utils/trans_config_utils.h"

namespace kv::test {
namespace {

void WriteTempConfig(const char* path, const std::string& content)
{
    std::ofstream configFile{path};
    ASSERT_TRUE(configFile.is_open());
    configFile << content;
}

}  // namespace

// ─── 设计默认值（等价类：合法默认） ───

TEST(TransConfigUtilsTest, StructuredConfigDefaultsMatchDesign)
{
    const TransportConfig config;

    ASSERT_TRUE(config.retry.Validate().ok());
    EXPECT_EQ(config.retry.busyRetryCount, std::uint32_t{2});
    EXPECT_EQ(config.retry.busyRetryBackoffBaseMs, std::uint32_t{1});
    EXPECT_EQ(config.retry.busyRetryBackoffMaxMs, std::uint32_t{100});
    EXPECT_EQ(config.retry.failoverRetryCount, std::uint32_t{2});
    EXPECT_EQ(config.retry.taskTimeoutMs, std::uint64_t{100});
    EXPECT_EQ(config.retry.attemptTimeoutMs, std::uint32_t{100});

    ASSERT_TRUE(config.recovery.Validate().ok());
    EXPECT_EQ(config.recovery.connectTimeoutMs, std::uint32_t{5000});
    EXPECT_EQ(config.recovery.connectRetryPerRound, std::uint32_t{2});
    EXPECT_EQ(config.recovery.recoverInitialBackoffMs, std::uint32_t{100});
    EXPECT_EQ(config.recovery.recoverMaxBackoffMs, std::uint32_t{5000});
    EXPECT_EQ(config.recovery.recoverCheckIntervalMs, std::uint32_t{100});

    ASSERT_TRUE(config.probe.Validate().ok());
    EXPECT_TRUE(config.probe.probeEnabled);
    EXPECT_EQ(config.probe.probePeriodMs, std::uint32_t{1000});
    EXPECT_EQ(config.probe.probeRetryCount, std::uint32_t{0});
    EXPECT_EQ(config.probe.probeRetryBackoffMs, std::uint32_t{5});
    EXPECT_EQ(config.probe.probeTimeoutMs, std::uint32_t{1000});
    EXPECT_EQ(config.probe.maxConcurrentTransports, std::uint32_t{4});
    EXPECT_EQ(config.probe.maxChannelsPerTransport, std::uint32_t{16});

    ASSERT_TRUE(config.flowControl.Validate().ok());
    EXPECT_FALSE(config.flowControl.flowControlEnabled);
    EXPECT_EQ(config.flowControl.maxInflightPerChannel, std::uint32_t{256});
    EXPECT_EQ(config.flowControl.busyFloor, std::uint32_t{32});
    EXPECT_EQ(config.flowControl.stableSuccessMs, std::uint64_t{60000});

    ASSERT_TRUE(config.connection.Validate().ok());
    EXPECT_EQ(config.connection.selectionPolicy, SelectionPolicy::ROUND_ROBIN);
    // 统一 X=1：单次连续链路失败即排空。
    EXPECT_EQ(config.connection.maxErrorCount, std::uint32_t{1});
}

// ─── RetryConfig 边界值分析（0..16 与 >0 字段） ───

TEST(TransConfigUtilsTest, RetryConfigBoundaries)
{
    {
        RetryConfig cfg;  // 0 是合法下界（关闭该类重试）
        cfg.busyRetryCount = 0;
        cfg.failoverRetryCount = 0;
        EXPECT_TRUE(cfg.Validate().ok());
    }
    {
        RetryConfig cfg;  // 16 是合法上界
        cfg.busyRetryCount = 16;
        cfg.failoverRetryCount = 16;
        EXPECT_TRUE(cfg.Validate().ok());
    }
    {
        RetryConfig cfg;  // 17 越界
        cfg.busyRetryCount = 17;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
    {
        RetryConfig cfg;
        cfg.failoverRetryCount = 17;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
    {
        RetryConfig cfg;  // >0 字段的 0 值非法
        cfg.taskTimeoutMs = 0;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
    {
        RetryConfig cfg;
        cfg.attemptTimeoutMs = 0;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
    {
        RetryConfig cfg;
        cfg.busyRetryBackoffBaseMs = 0;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
}

// ─── RecoveryConfig 边界值分析 ───

TEST(TransConfigUtilsTest, RecoveryConfigBoundaries)
{
    {
        RecoveryConfig cfg;  // K=0 合法（单次建链）
        cfg.connectRetryPerRound = 0;
        EXPECT_TRUE(cfg.Validate().ok());
    }
    {
        RecoveryConfig cfg;
        cfg.connectRetryPerRound = 16;
        EXPECT_TRUE(cfg.Validate().ok());
    }
    {
        RecoveryConfig cfg;
        cfg.connectRetryPerRound = 17;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
    {
        RecoveryConfig cfg;
        cfg.recoverInitialBackoffMs = 0;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
    {
        RecoveryConfig cfg;
        cfg.recoverCheckIntervalMs = 0;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
}

// ─── ProbeConfig 边界值分析 + 派生上限 ───

TEST(TransConfigUtilsTest, ProbeConfigBoundariesAndDerivedLimit)
{
    {
        ProbeConfig cfg;
        cfg.maxConcurrentTransports = 1;
        cfg.maxChannelsPerTransport = 1;
        EXPECT_TRUE(cfg.Validate().ok());
        EXPECT_EQ(cfg.DerivedInFlightLimit(), std::uint64_t{1});
    }
    {
        ProbeConfig cfg;  // 上界 1024 x 4096 = 4M
        cfg.maxConcurrentTransports = 1024;
        cfg.maxChannelsPerTransport = 4096;
        EXPECT_TRUE(cfg.Validate().ok());
        EXPECT_EQ(cfg.DerivedInFlightLimit(), std::uint64_t{1024} * 4096);
    }
    {
        ProbeConfig cfg;
        cfg.maxConcurrentTransports = 1025;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
    {
        ProbeConfig cfg;
        cfg.maxConcurrentTransports = 0;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
    {
        ProbeConfig cfg;
        cfg.maxChannelsPerTransport = 4097;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
    {
        ProbeConfig cfg;
        cfg.probeRetryCount = 16;
        EXPECT_TRUE(cfg.Validate().ok());
    }
    {
        ProbeConfig cfg;
        cfg.probeRetryCount = 17;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
    {
        ProbeConfig cfg;
        cfg.probePeriodMs = 0;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
    {
        ProbeConfig cfg;
        cfg.probeTimeoutMs = 0;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
}

// ─── FlowControlConfig 关系约束（busyFloor 关系边界） ───

TEST(TransConfigUtilsTest, FlowControlBusyFloorRelationBoundaries)
{
    {
        FlowControlConfig cfg;  // busyFloor=1 是合法下界（需 max>1）
        cfg.busyFloor = 1;
        EXPECT_TRUE(cfg.Validate().ok());
    }
    {
        FlowControlConfig cfg;
        cfg.busyFloor = 255;
        cfg.maxInflightPerChannel = 256;
        EXPECT_TRUE(cfg.Validate().ok());  // T = max-1 合法
    }
    {
        FlowControlConfig cfg;
        cfg.busyFloor = 256;
        cfg.maxInflightPerChannel = 256;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);  // T == max 非法
    }
    {
        FlowControlConfig cfg;
        cfg.busyFloor = 300;
        cfg.maxInflightPerChannel = 256;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);  // T > max 非法
    }
    {
        FlowControlConfig cfg;
        cfg.busyFloor = 0;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);  // T = 0 非法
    }
    {
        FlowControlConfig cfg;
        cfg.stableSuccessMs = 0;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
}

// ─── ConnectionConfig 边界（X >= 1） ───

TEST(TransConfigUtilsTest, ConnectionConfigMaxErrorCountBoundaries)
{
    {
        ConnectionConfig cfg;
        cfg.maxErrorCount = 1;
        EXPECT_TRUE(cfg.Validate().ok());
    }
    {
        ConnectionConfig cfg;
        cfg.maxErrorCount = 0;
        EXPECT_EQ(cfg.Validate().code, StatusCode::INVALID_ARGUMENT);
    }
    {
        ConnectionConfig cfg;  // 显式阈值可大于 1（兼容配置）
        cfg.maxErrorCount = 3;
        EXPECT_TRUE(cfg.Validate().ok());
    }
}

// ─── LoadTransportConfig 文件解析（结构化 section 键） ───

TEST(TransConfigUtilsTest, LoadTransportConfigParsesStructuredSections)
{
    constexpr const char* kConfigPath = "trans_config_structured_ok_test.conf";
    WriteTempConfig(kConfigPath,
                    "kernel_count=1\n"
                    "quiet_count=1\n"
                    "retry.busy_retry_count=5\n"
                    "retry.busy_retry_backoff_base_ms=2\n"
                    "retry.busy_retry_backoff_max_ms=50\n"
                    "retry.failover_retry_count=1\n"
                    "retry.task_timeout_ms=250\n"
                    "retry.attempt_timeout_ms=80\n"
                    "recovery.connect_timeout_ms=1500\n"
                    "recovery.connect_retry_per_round=4\n"
                    "recovery.recover_initial_backoff_ms=60\n"
                    "recovery.recover_max_backoff_ms=900\n"
                    "recovery.recover_check_interval_ms=30\n"
                    "probe.enabled=false\n"
                    "probe.period_ms=700\n"
                    "probe.retry_count=1\n"
                    "probe.retry_backoff_ms=4\n"
                    "probe.timeout_ms=600\n"
                    "probe.max_concurrent_transports=8\n"
                    "probe.max_channels_per_transport=32\n"
                    "flow_control.enabled=true\n"
                    "flow_control.max_inflight_per_channel=128\n"
                    "flow_control.busy_floor=16\n"
                    "flow_control.stable_success_ms=30000\n"
                    "connection.selection_policy=LEAST_LOADED\n"
                    "connection.max_error_count=2\n");

    TransportConfig config;
    const auto status = LoadTransportConfig(kConfigPath, config);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(config.retry.busyRetryCount, std::uint32_t{5});
    EXPECT_EQ(config.retry.busyRetryBackoffBaseMs, std::uint32_t{2});
    EXPECT_EQ(config.retry.busyRetryBackoffMaxMs, std::uint32_t{50});
    EXPECT_EQ(config.retry.failoverRetryCount, std::uint32_t{1});
    EXPECT_EQ(config.retry.taskTimeoutMs, std::uint64_t{250});
    EXPECT_EQ(config.retry.attemptTimeoutMs, std::uint32_t{80});
    EXPECT_EQ(config.recovery.connectTimeoutMs, std::uint32_t{1500});
    EXPECT_EQ(config.recovery.connectRetryPerRound, std::uint32_t{4});
    EXPECT_EQ(config.recovery.recoverInitialBackoffMs, std::uint32_t{60});
    EXPECT_EQ(config.recovery.recoverMaxBackoffMs, std::uint32_t{900});
    EXPECT_EQ(config.recovery.recoverCheckIntervalMs, std::uint32_t{30});
    EXPECT_FALSE(config.probe.probeEnabled);
    EXPECT_EQ(config.probe.probePeriodMs, std::uint32_t{700});
    EXPECT_EQ(config.probe.probeRetryCount, std::uint32_t{1});
    EXPECT_EQ(config.probe.probeRetryBackoffMs, std::uint32_t{4});
    EXPECT_EQ(config.probe.probeTimeoutMs, std::uint32_t{600});
    EXPECT_EQ(config.probe.maxConcurrentTransports, std::uint32_t{8});
    EXPECT_EQ(config.probe.maxChannelsPerTransport, std::uint32_t{32});
    EXPECT_TRUE(config.flowControl.flowControlEnabled);
    EXPECT_EQ(config.flowControl.maxInflightPerChannel, std::uint32_t{128});
    EXPECT_EQ(config.flowControl.busyFloor, std::uint32_t{16});
    EXPECT_EQ(config.flowControl.stableSuccessMs, std::uint64_t{30000});
    EXPECT_EQ(config.connection.selectionPolicy, SelectionPolicy::LEAST_LOADED);
    EXPECT_EQ(config.connection.maxErrorCount, std::uint32_t{2});
}

// 越界值经 LoadTransportConfig 校验为 INVALID_ARGUMENT（禁止静默回退）。
TEST(TransConfigUtilsTest, LoadTransportConfigRejectsOutOfRangeValues)
{
    constexpr const char* kConfigPath = "trans_config_out_of_range_test.conf";
    WriteTempConfig(kConfigPath, "retry.busy_retry_count=17\n");

    TransportConfig config;
    const auto status = LoadTransportConfig(kConfigPath, config);
    std::remove(kConfigPath);

    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
}

// 非法文本（无法解析）不回退默认值，而是标记后由校验拒绝。
TEST(TransConfigUtilsTest, LoadTransportConfigRejectsGarbageValues)
{
    constexpr const char* kConfigPath = "trans_config_garbage_test.conf";
    WriteTempConfig(kConfigPath, "retry.task_timeout_ms=not-a-number\n");

    TransportConfig config;
    const auto status = LoadTransportConfig(kConfigPath, config);
    std::remove(kConfigPath);

    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
}

// 关系约束经文件入口同样生效。
TEST(TransConfigUtilsTest, LoadTransportConfigRejectsBusyFloorRelation)
{
    constexpr const char* kConfigPath = "trans_config_busy_floor_test.conf";
    WriteTempConfig(kConfigPath,
                    "flow_control.max_inflight_per_channel=64\n"
                    "flow_control.busy_floor=64\n");

    TransportConfig config;
    const auto status = LoadTransportConfig(kConfigPath, config);
    std::remove(kConfigPath);

    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
}

// legacy 键同步：max_error_count 同时落到 connection.maxErrorCount。
TEST(TransConfigUtilsTest, LoadTransportConfigSyncsLegacyMaxErrorCount)
{
    constexpr const char* kConfigPath = "trans_config_legacy_error_count_test.conf";
    WriteTempConfig(kConfigPath, "kernel_count=1\nquiet_count=1\nmax_error_count=5\n");

    TransportConfig config;
    const auto status = LoadTransportConfig(kConfigPath, config);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(config.maxErrorCount, std::uint32_t{5});
    EXPECT_EQ(config.connection.maxErrorCount, std::uint32_t{5});
}

// ─── 客户端入口（transport. 前缀键） ───

TEST(TransConfigUtilsTest, ClientConfigParsesTransportSections)
{
    constexpr const char* kConfigPath = "kv_client_structured_sections_test.conf";
    WriteTempConfig(kConfigPath,
                    "wait_timeout_ms=999\n"
                    "transport.asu_ids=10\n"
                    "transport.retry.task_timeout_ms=321\n"
                    "transport.retry.failover_retry_count=4\n"
                    "transport.connection.max_error_count=2\n"
                    "transport.flow_control.enabled=true\n"
                    "transport.flow_control.busy_floor=8\n"
                    "transport.probe.enabled=false\n"
                    "transport.probe.period_ms=500\n");

    KvClientConfig config;
    const auto status = KvClientConfig::LoadFromFile(kConfigPath, config);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(config.transportConfigs.size(), std::size_t{1});
    const auto& transport = config.transportConfigs.front();
    // 显式 transport.retry.task_timeout_ms 覆盖 wait_timeout_ms 镜像。
    EXPECT_EQ(transport.retry.taskTimeoutMs, std::uint64_t{321});
    EXPECT_EQ(transport.timeoutMs, std::uint64_t{999});
    EXPECT_EQ(transport.retry.failoverRetryCount, std::uint32_t{4});
    EXPECT_EQ(transport.connection.maxErrorCount, std::uint32_t{2});
    EXPECT_TRUE(transport.flowControl.flowControlEnabled);
    EXPECT_EQ(transport.flowControl.busyFloor, std::uint32_t{8});
    EXPECT_FALSE(transport.probe.probeEnabled);
    EXPECT_EQ(transport.probe.probePeriodMs, std::uint32_t{500});
}

// 未显式配置时，transport 任务总 Deadline 镜像 client wait 超时（兼容语义）。
TEST(TransConfigUtilsTest, ClientConfigMirrorsWaitTimeoutIntoTaskDeadline)
{
    constexpr const char* kConfigPath = "kv_client_mirror_deadline_test.conf";
    WriteTempConfig(kConfigPath, "wait_timeout_ms=777\ntransport.asu_ids=10\n");

    KvClientConfig config;
    const auto status = KvClientConfig::LoadFromFile(kConfigPath, config);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(config.transportConfigs.size(), std::size_t{1});
    EXPECT_EQ(config.transportConfigs.front().retry.taskTimeoutMs, std::uint64_t{777});
}

// legacy transport.max_error_count 在客户端入口同样同步结构化字段。
TEST(TransConfigUtilsTest, ClientConfigSyncsLegacyMaxErrorCount)
{
    constexpr const char* kConfigPath = "kv_client_legacy_error_count_test.conf";
    WriteTempConfig(kConfigPath,
                    "transport.asu_ids=10\n"
                    "transport.max_error_count=6\n");

    KvClientConfig config;
    const auto status = KvClientConfig::LoadFromFile(kConfigPath, config);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(config.transportConfigs.size(), std::size_t{1});
    EXPECT_EQ(config.transportConfigs.front().maxErrorCount, std::uint32_t{6});
    EXPECT_EQ(config.transportConfigs.front().connection.maxErrorCount, std::uint32_t{6});
}

}  // namespace kv::test
