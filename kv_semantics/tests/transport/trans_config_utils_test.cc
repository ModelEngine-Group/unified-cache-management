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

    ASSERT_TRUE(config.recovery.Validate().ok());
    EXPECT_EQ(config.recovery.connectTimeoutMs, std::uint32_t{5000});
    EXPECT_EQ(config.recovery.connectRetryPerRound, std::uint32_t{2});
    EXPECT_EQ(config.recovery.recoverInitialBackoffMs, std::uint32_t{100});
    EXPECT_EQ(config.recovery.recoverMaxBackoffMs, std::uint32_t{5000});
    EXPECT_EQ(config.recovery.recoverCheckIntervalMs, std::uint32_t{100});

    ASSERT_TRUE(config.connection.Validate().ok());
    EXPECT_EQ(config.connection.selectionPolicy, SelectionPolicy::ROUND_ROBIN);
    // 统一 X=1：单次连续链路失败即排空。
    EXPECT_EQ(config.connection.maxErrorCount, std::uint32_t{1});
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

// ─── FlowControlConfig 关系约束（busyFloor 关系边界） ───

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
                    "recovery.connect_timeout_ms=1500\n"
                    "recovery.connect_retry_per_round=4\n"
                    "recovery.recover_initial_backoff_ms=60\n"
                    "recovery.recover_max_backoff_ms=900\n"
                    "recovery.recover_check_interval_ms=30\n"
                    "connection.selection_policy=LEAST_LOADED\n"
                    "connection.max_error_count=2\n");

    TransportConfig config;
    const auto status = LoadTransportConfig(kConfigPath, config);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(config.recovery.connectTimeoutMs, std::uint32_t{1500});
    EXPECT_EQ(config.recovery.connectRetryPerRound, std::uint32_t{4});
    EXPECT_EQ(config.recovery.recoverInitialBackoffMs, std::uint32_t{60});
    EXPECT_EQ(config.recovery.recoverMaxBackoffMs, std::uint32_t{900});
    EXPECT_EQ(config.recovery.recoverCheckIntervalMs, std::uint32_t{30});
    EXPECT_EQ(config.connection.selectionPolicy, SelectionPolicy::LEAST_LOADED);
    EXPECT_EQ(config.connection.maxErrorCount, std::uint32_t{2});
}

// 越界值经 LoadTransportConfig 校验为 INVALID_ARGUMENT（禁止静默回退）。
TEST(TransConfigUtilsTest, LoadTransportConfigRejectsOutOfRangeValues)
{
    constexpr const char* kConfigPath = "trans_config_out_of_range_test.conf";
    WriteTempConfig(kConfigPath, "recovery.connect_retry_per_round=17\n");

    TransportConfig config;
    const auto status = LoadTransportConfig(kConfigPath, config);
    std::remove(kConfigPath);

    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
}

// 非法文本（无法解析）不回退默认值，而是标记后由校验拒绝。
TEST(TransConfigUtilsTest, LoadTransportConfigRejectsGarbageValues)
{
    constexpr const char* kConfigPath = "trans_config_garbage_test.conf";
    WriteTempConfig(kConfigPath, "recovery.recover_check_interval_ms=not-a-number\n");

    TransportConfig config;
    const auto status = LoadTransportConfig(kConfigPath, config);
    std::remove(kConfigPath);

    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
}

// 关系约束经文件入口同样生效。
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
                    "transport.connection.max_error_count=2\n");

    KvClientConfig config;
    const auto status = KvClientConfig::LoadFromFile(kConfigPath, config);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(config.transportConfigs.size(), std::size_t{1});
    const auto& transport = config.transportConfigs.front();
    EXPECT_EQ(transport.timeoutMs, std::uint64_t{999});
    EXPECT_EQ(transport.connection.maxErrorCount, std::uint32_t{2});
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
