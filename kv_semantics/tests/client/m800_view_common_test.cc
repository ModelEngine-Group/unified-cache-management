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

TEST(M800ViewCommonTest, BuildSupportsPromptAndNormalSpellings)
{
    auto legacyConfig = MakeWorkerConfig();
    legacyConfig.attrs["view.refresh_in_terval_ms"] = "41000";
    legacyConfig.attrs["view.agent_fetch_timeoout_ms"] = "11000";
    legacyConfig.attrs["view.endpoint_protocol"] = "RoCe";
    legacyConfig.attrs["view.endpoint_port"] = "7000";
    M800ViewServerConfig legacy;

    ASSERT_TRUE(M800ViewServerConfig::Build(legacyConfig, legacy).ok());
    EXPECT_EQ(legacy.refreshIntervalMs, std::uint64_t{41000});
    EXPECT_EQ(legacy.agentFetchTimeoutMs, std::uint64_t{11000});
    EXPECT_EQ(legacy.endpointProtocol, Protocol::ROCE);
    EXPECT_EQ(legacy.endpointPort, std::uint16_t{7000});

    auto normalConfig = MakeWorkerConfig();
    normalConfig.attrs["view.refresh_interval_ms"] = "42000";
    normalConfig.attrs["view.refresh_in_terval_ms"] = "41000";
    normalConfig.attrs["view.agent_fetch_timeout_ms"] = "12000";
    normalConfig.attrs["view.agent_fetch_timeoout_ms"] = "11000";
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
}

TEST(M800ViewCommonTest, BuildInvalidNumbersAndProtocolFallBackToDefaults)
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

TEST(M800ViewCommonTest, BuildOutOfRangeTimeoutsFallBackToDefaults)
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
