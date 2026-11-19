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
#include "view/m800_view_common.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace kv::test {
namespace {

KvClientConfig MakeWorkerConfig()
{
    KvClientConfig config;
    config.attrs["view.role"] = "worker";
    config.attrs["view.master_addrs"] = "127.0.0.1:9100";
    return config;
}

// 测试场景：使用空串、普通字符串、不同长度输入、内嵌 NUL 及已知冲突池名作为 M800PoolNameToKvNsId 的输入。
// 预期结果：所有结果逐一等于标准 XXH32 测试向量，包括内嵌 NUL 输入和两个 pool 名称得到相同哈希值的情况。
TEST(M800ViewCommonTest, PoolNameHashMatchesStandardXxh32Vectors)
{
    struct HashVector {
        std::string input;
        std::uint32_t expected;
    };
    const std::vector<HashVector> vectors{
        {"", 0xA3316A09U},
        {"a", 0xC55804D1U},
        {"abc", 0x6E40A53AU},
        {"123456789012345", 0xC42C867AU},
        {"1234567890123456", 0xF8EB1468U},
        {"12345678901234567", 0x096704CDU},
        {std::string{"a\0b", 3}, 0xBE1EDC77U},
        {"pool-69670", 0x0B33F062U},
        {"pool-261455", 0x0B33F062U},
    };

    for (std::size_t index = 0; index < vectors.size(); ++index) {
        SCOPED_TRACE("hash vector index=" + std::to_string(index));
        EXPECT_EQ(M800PoolNameToKvNsId(vectors[index].input), vectors[index].expected);
    }
}

// 测试场景：仅设置 view.ceph.mon_host，未指定角色及其余配置时构建服务端配置。
// 预期结果：构建成功，角色为 MASTER，监听 0.0.0.0:9100，Master 地址为空，monHost 为 127.0.0.1:6789，其余 Ceph/agent 默认值、10000/2000/30000 ms 时间参数及 UB/6000 端点均正确。
TEST(M800ViewCommonTest, BuildUsesMasterDefaults)
{
    KvClientConfig config;
    config.attrs["view.ceph.mon_host"] = "127.0.0.1:6789";
    M800ViewServerConfig built;

    const auto status = M800ViewServerConfig::Build(config, built);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(built.role, M800ViewRole::MASTER);
    EXPECT_EQ(built.listenAddr, "0.0.0.0:9100");
    EXPECT_TRUE(built.masterAddr.empty());
    EXPECT_TRUE(built.masterAddrs.empty());
    EXPECT_EQ(built.monHost, "127.0.0.1:6789");
    EXPECT_EQ(built.clientName, "admin");
    EXPECT_EQ(built.keyringPath, "/etc/ceph/ceph.client.admin.keyring");
    EXPECT_EQ(built.agentLibPath, "/usr/lib/libkv_agent.so");
    EXPECT_EQ(built.agentFetchTimeoutMs, std::uint64_t{10000});
    EXPECT_EQ(built.httpTimeoutMs, std::uint64_t{2000});
    EXPECT_EQ(built.refreshIntervalMs, std::uint64_t{30000});
    EXPECT_EQ(built.endpointProtocol, Protocol::UB);
    EXPECT_EQ(built.endpointPort, std::uint16_t{6000});
}

// 测试场景：以混合大小写和空白设置 worker 角色，配置含 IPv4/IPv6 的 master_addrs，并预置旧 clientKey 与 worker keyring 路径。
// 预期结果：构建成功并识别为 SLAVE，地址去空白后按原顺序解析且首个写入 masterAddr，monHost 为空，不读取 keyring 且 clientKey 被清空。
TEST(M800ViewCommonTest, BuildWorkerParsesAddressesWithoutReadingKeyring)
{
    KvClientConfig config;
    config.attrs["view.role"] = " WoRkEr ";
    config.attrs["view.master_addrs"] = " 127.0.0.1:9100, [::1]:9100 ";
    config.attrs["view.ceph.keyring"] = "worker-keyring-must-not-be-read";
    M800ViewServerConfig built;
    built.clientKey = "stale-secret";

    const auto status = M800ViewServerConfig::Build(config, built);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(built.role, M800ViewRole::SLAVE);
    EXPECT_EQ(built.masterAddrs,
              (std::vector<std::string>{"127.0.0.1:9100", "[::1]:9100"}));
    EXPECT_EQ(built.masterAddr, "127.0.0.1:9100");
    EXPECT_TRUE(built.monHost.empty());
    EXPECT_TRUE(built.clientKey.empty());
}

// 测试场景：配置 view.role=slave 构建服务端配置，输出对象预置旧角色、地址、密钥和 TCP 端点。
// 预期结果：返回 INVALID_ARGUMENT，并清除旧输出，恢复 MASTER、0.0.0.0:9100、空 Master 地址/密钥、admin 及 UB/6000 等默认值。
TEST(M800ViewCommonTest, BuildRejectsInvalidRoleAndClearsPreviousOutput)
{
    KvClientConfig config;
    config.attrs["view.role"] = "slave";
    M800ViewServerConfig built;
    built.role = M800ViewRole::SLAVE;
    built.listenAddr = "old-listen";
    built.masterAddr = "old-master";
    built.masterAddrs = {"old-master"};
    built.clientKey = "old-secret";
    built.endpointProtocol = Protocol::TCP;
    built.endpointPort = 1;

    const auto status = M800ViewServerConfig::Build(config, built);

    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(built.role, M800ViewRole::MASTER);
    EXPECT_EQ(built.listenAddr, "0.0.0.0:9100");
    EXPECT_TRUE(built.masterAddr.empty());
    EXPECT_TRUE(built.masterAddrs.empty());
    EXPECT_TRUE(built.clientKey.empty());
    EXPECT_EQ(built.clientName, "admin");
    EXPECT_EQ(built.endpointProtocol, Protocol::UB);
    EXPECT_EQ(built.endpointPort, std::uint16_t{6000});
}

// 测试场景：Master 只有 legacy viewServiceAddrs 而缺少必需 Ceph 配置，以及 Worker 的 master_addrs 仅含逗号和空白时分别构建配置。
// 预期结果：两种缺少角色必需字段的配置均返回 INVALID_ARGUMENT，legacy viewServiceAddrs 不能替代 Master 必需配置。
TEST(M800ViewCommonTest, BuildEnforcesRoleSpecificRequiredFields)
{
    KvClientConfig masterConfig;
    masterConfig.viewServiceAddrs = {"legacy-address-must-not-be-used"};
    M800ViewServerConfig built;
    EXPECT_EQ(M800ViewServerConfig::Build(masterConfig, built).code,
              StatusCode::INVALID_ARGUMENT);

    auto workerConfig = MakeWorkerConfig();
    workerConfig.attrs["view.master_addrs"] = " , ,";
    EXPECT_EQ(M800ViewServerConfig::Build(workerConfig, built).code,
              StatusCode::INVALID_ARGUMENT);
}

// 测试场景：Master 指向含注释、section 和重复 key 行的 keyring，Worker 也配置同一路径。
// 预期结果：两者构建均成功；Master 仅读取首个 key 的值 AQIDBAU=++== 并忽略后续重复项，Worker 不读取文件且 clientKey 为空。
TEST(M800ViewCommonTest, BuildReadsFirstKeyValueOnlyForMaster)
{
    const std::string keyringPath = "m800_view_common_test_keyring.conf";
    {
        std::ofstream keyring{keyringPath};
        ASSERT_TRUE(keyring.is_open());
        keyring << "  # comment\n"
                << "  ; comment\n"
                << "[client.admin]\n"
                << "  key = AQIDBAU=++==  \n"
                << "  key = ignored\n";
    }

    KvClientConfig masterConfig;
    masterConfig.attrs["view.ceph.mon_host"] = "mon1:6789";
    masterConfig.attrs["view.ceph.keyring"] = keyringPath;
    M800ViewServerConfig master;
    const auto masterStatus = M800ViewServerConfig::Build(masterConfig, master);

    auto workerConfig = MakeWorkerConfig();
    workerConfig.attrs["view.ceph.keyring"] = keyringPath;
    M800ViewServerConfig worker;
    const auto workerStatus = M800ViewServerConfig::Build(workerConfig, worker);
    std::remove(keyringPath.c_str());

    ASSERT_TRUE(masterStatus.ok()) << masterStatus.message;
    EXPECT_EQ(master.clientKey, "AQIDBAU=++==");
    ASSERT_TRUE(workerStatus.ok()) << workerStatus.message;
    EXPECT_TRUE(worker.clientKey.empty());
}

// 测试场景：Master 配置不存在的 keyring 文件，且输出对象预置 clientKey 为 stale-secret。
// 预期结果：构建仍返回成功，旧 clientKey 被清空。
TEST(M800ViewCommonTest, BuildUnreadableKeyringReturnsOkWithEmptyClientKey)
{
    KvClientConfig config;
    config.attrs["view.ceph.mon_host"] = "mon1:6789";
    config.attrs["view.ceph.keyring"] = "m800-view-common-file-that-does-not-exist";
    M800ViewServerConfig built;
    built.clientKey = "stale-secret";

    const auto status = M800ViewServerConfig::Build(config, built);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_TRUE(built.clientKey.empty());
}

// 测试场景：Worker 仅通过旧配置项 view.master_addr 指定 Master 地址。
// 预期结果：构建成功，masterAddrs 仅含 127.0.0.1:9100，masterAddr 也为该地址。
TEST(M800ViewCommonTest, BuildSupportsLegacyMasterAddr)
{
    KvClientConfig config;
    config.attrs["view.role"] = "worker";
    config.attrs["view.master_addr"] = "127.0.0.1:9100";
    M800ViewServerConfig built;

    ASSERT_TRUE(M800ViewServerConfig::Build(config, built).ok());
    EXPECT_EQ(built.masterAddrs, std::vector<std::string>{"127.0.0.1:9100"});
    EXPECT_EQ(built.masterAddr, "127.0.0.1:9100");
}

// 测试场景：Worker 同时配置旧 master_addr 和包含两个地址的新 master_addrs。
// 预期结果：构建成功并优先使用新配置，masterAddrs 为两个新地址且 masterAddr 取首个 127.0.0.2:9100。
TEST(M800ViewCommonTest, BuildPrefersMasterAddrsOverLegacyMasterAddr)
{
    KvClientConfig config;
    config.attrs["view.role"] = "worker";
    config.attrs["view.master_addr"] = "127.0.0.1:9100";
    config.attrs["view.master_addrs"] = "127.0.0.2:9100,127.0.0.3:9100";
    M800ViewServerConfig built;

    ASSERT_TRUE(M800ViewServerConfig::Build(config, built).ok());
    EXPECT_EQ(built.masterAddrs,
              (std::vector<std::string>{"127.0.0.2:9100", "127.0.0.3:9100"}));
    EXPECT_EQ(built.masterAddr, "127.0.0.2:9100");
}

// 测试场景：Worker 同时设置有效的旧 master_addr 和显式为空的新 master_addrs。
// 预期结果：空的新配置仍优先于旧配置，构建返回 INVALID_ARGUMENT，不回退使用旧地址。
TEST(M800ViewCommonTest, BuildMasterAddrsEmptyTakesPrecedenceOverLegacyMasterAddr)
{
    KvClientConfig config;
    config.attrs["view.role"] = "worker";
    config.attrs["view.master_addr"] = "127.0.0.1:9100";
    config.attrs["view.master_addrs"] = "";
    M800ViewServerConfig built;

    EXPECT_EQ(M800ViewServerConfig::Build(config, built).code, StatusCode::INVALID_ARGUMENT);
}

// 测试场景：Worker 分别使用标准时间/端点配置键，以及同时设置 view.ceph.agent_fetch_timeout_ms 和标准 agent 超时键。
// 预期结果：标准键值 42000、12000、2500、TCP、65535 被正确解析；Ceph 前缀键优先，使 agent 获取超时为 13000 ms。
TEST(M800ViewCommonTest, BuildSupportsCanonicalAndCephTimeoutKeys)
{
    auto normalConfig = MakeWorkerConfig();
    normalConfig.attrs["view.refresh_interval_ms"] = "42000";
    normalConfig.attrs["view.agent_fetch_timeout_ms"] = "12000";
    normalConfig.attrs["view.http_timeout_ms"] = "2500";
    normalConfig.attrs["view.endpoint_protocol"] = "TCP";
    normalConfig.attrs["view.endpoint_port"] = "65535";
    M800ViewServerConfig normal;

    ASSERT_TRUE(M800ViewServerConfig::Build(normalConfig, normal).ok());
    EXPECT_EQ(normal.refreshIntervalMs, std::uint64_t{42000});
    EXPECT_EQ(normal.agentFetchTimeoutMs, std::uint64_t{12000});
    EXPECT_EQ(normal.httpTimeoutMs, std::uint64_t{2500});
    EXPECT_EQ(normal.endpointProtocol, Protocol::TCP);
    EXPECT_EQ(normal.endpointPort, std::uint16_t{65535});

    auto cephConfig = MakeWorkerConfig();
    cephConfig.attrs["view.ceph.agent_fetch_timeout_ms"] = "13000";
    cephConfig.attrs["view.agent_fetch_timeout_ms"] = "12000";
    M800ViewServerConfig ceph;
    ASSERT_TRUE(M800ViewServerConfig::Build(cephConfig, ceph).ok());
    EXPECT_EQ(ceph.agentFetchTimeoutMs, std::uint64_t{13000});
}

// 测试场景：Worker 将 view.endpoint_port 配置为 0。
// 预期结果：构建成功，端点端口回退为默认值 6000。
TEST(M800ViewCommonTest, BuildZeroEndpointPortFallsBackToDefault)
{
    auto config = MakeWorkerConfig();
    config.attrs["view.endpoint_port"] = "0";
    M800ViewServerConfig built;

    ASSERT_TRUE(M800ViewServerConfig::Build(config, built).ok());
    EXPECT_EQ(built.endpointPort, std::uint16_t{6000});
}

// 测试场景：Worker 使用拼错的 agent_fetch_timeoout_ms 和 refresh_in_terval_ms 配置键。
// 预期结果：构建成功并忽略这两个键，agent 获取超时和刷新间隔分别保持默认值 10000 ms 和 30000 ms。
TEST(M800ViewCommonTest, BuildIgnoresMisspelledTimeoutKeys)
{
    auto config = MakeWorkerConfig();
    config.attrs["view.agent_fetch_timeoout_ms"] = "11000";
    config.attrs["view.refresh_in_terval_ms"] = "41000";
    M800ViewServerConfig built;

    ASSERT_TRUE(M800ViewServerConfig::Build(config, built).ok());
    EXPECT_EQ(built.agentFetchTimeoutMs, std::uint64_t{10000});
    EXPECT_EQ(built.refreshIntervalMs, std::uint64_t{30000});
}

// 测试场景：Worker 配置带单位、负数或非数字的超时值，以及未知协议和超范围端口 65536。
// 预期结果：构建成功，三项时间配置依次回退为 30000、10000、2000 ms，端点回退为 UB 协议和 6000 端口。
TEST(M800ViewCommonTest, BuildInvalidNumbersAndProtocolFallbackToDefaults)
{
    auto config = MakeWorkerConfig();
    config.attrs["view.refresh_interval_ms"] = "12ms";
    config.attrs["view.agent_fetch_timeout_ms"] = "-1";
    config.attrs["view.http_timeout_ms"] = "bad-timeout";
    config.attrs["view.endpoint_protocol"] = "not-a-protocol";
    config.attrs["view.endpoint_port"] = "65536";
    config.attrs["view.ceph.filter"] = "up";
    M800ViewServerConfig built;

    const auto status = M800ViewServerConfig::Build(config, built);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(built.refreshIntervalMs, std::uint64_t{30000});
    EXPECT_EQ(built.agentFetchTimeoutMs, std::uint64_t{10000});
    EXPECT_EQ(built.httpTimeoutMs, std::uint64_t{2000});
    EXPECT_EQ(built.endpointProtocol, Protocol::UB);
    EXPECT_EQ(built.endpointPort, std::uint16_t{6000});
}

// 测试场景：Worker 的刷新间隔、agent/HTTP 超时和端点端口字符串均在合法数字后内嵌 NUL 与 junk。
// 预期结果：构建成功但不接受被 NUL 污染的数值，四项分别回退为 30000、10000、2000 ms 和 6000 端口。
TEST(M800ViewCommonTest, BuildRejectsEmbeddedNulInNumericValues)
{
    auto config = MakeWorkerConfig();
    config.attrs["view.refresh_interval_ms"] = std::string("41000\0junk", 9);
    config.attrs["view.agent_fetch_timeout_ms"] = std::string("11000\0junk", 9);
    config.attrs["view.http_timeout_ms"] = std::string("2500\0junk", 8);
    config.attrs["view.endpoint_port"] = std::string("7000\0junk", 8);
    M800ViewServerConfig built;

    ASSERT_TRUE(M800ViewServerConfig::Build(config, built).ok());
    EXPECT_EQ(built.refreshIntervalMs, 30000);
    EXPECT_EQ(built.agentFetchTimeoutMs, 10000);
    EXPECT_EQ(built.httpTimeoutMs, 2000);
    EXPECT_EQ(built.endpointPort, 6000);
}

// 测试场景：分别向 view.ceph.mon_host 和 view.ceph.agent_lib 写入包含内嵌 NUL 的连接或路径字符串。
// 预期结果：两种配置均返回 INVALID_ARGUMENT。
TEST(M800ViewCommonTest, BuildRejectsEmbeddedNulInAgentConnectionStrings)
{
    for (const auto& key : {std::string("view.ceph.mon_host"), std::string("view.ceph.agent_lib")}) {
        auto config = MakeWorkerConfig();
        config.attrs[key] = std::string("bad\0value", 9);
        M800ViewServerConfig built;
        EXPECT_EQ(M800ViewServerConfig::Build(config, built).code, StatusCode::INVALID_ARGUMENT) << key;
    }
}

// 测试场景：Worker 将刷新间隔、agent 获取超时和 HTTP 超时都设为 uint64_t 最大值 18446744073709551615。
// 预期结果：构建成功，并因超出允许范围分别回退为 30000、10000 和 2000 ms。
TEST(M800ViewCommonTest, BuildOutOfRangeTimeoutsFallbackToDefaults)
{
    auto config = MakeWorkerConfig();
    config.attrs["view.refresh_interval_ms"] = "18446744073709551615";
    config.attrs["view.agent_fetch_timeout_ms"] = "18446744073709551615";
    config.attrs["view.http_timeout_ms"] = "18446744073709551615";
    M800ViewServerConfig built;

    ASSERT_TRUE(M800ViewServerConfig::Build(config, built).ok());
    EXPECT_EQ(built.refreshIntervalMs, 30000);
    EXPECT_EQ(built.agentFetchTimeoutMs, 10000);
    EXPECT_EQ(built.httpTimeoutMs, 2000);
}

}  // namespace
}  // namespace kv::test
