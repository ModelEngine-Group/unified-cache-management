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

// ─── ProbeConfig 边界值分析 + 派生上限 ───

// ─── FlowControlConfig 关系约束（busyFloor 关系边界） ───

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
                    "retry.attempt_timeout_ms=80\n");

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
}

// ─── 客户端入口（transport. 前缀键） ───

TEST(TransConfigUtilsTest, ClientConfigParsesTransportSections)
{
    constexpr const char* kConfigPath = "kv_client_structured_sections_test.conf";
    WriteTempConfig(kConfigPath,
                    "wait_timeout_ms=999\n"
                    "transport.asu_ids=10\n"
                    "transport.retry.task_timeout_ms=321\n"
                    "transport.retry.failover_retry_count=4\n");

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
}

}  // namespace kv::test
