#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include "parser_common.h"
#include "roce/cpu_kv_over_roce_trans_provider.h"

namespace kv::test {
namespace {

TransProvider::ConnectionHandle FakeHandle(std::uintptr_t value)
{
    return reinterpret_cast<void*>(value);
}

TransportConfig MakeRoceConfig()
{
    TransportConfig config;
    config.nodeName = "asu-1";
    config.nodeId = 1;
    config.qpNum = 2;
    config.endpoints.push_back(
        NodeEndpoint{"10.0.0.11", 19001, Protocol::ROCE, -1, "mlx5_0", 1, {}});
    // The provider requires the GID table index, so the fixture has to name one.
    config.attrs["roce_gid_index"] = "0";
    return config;
}

TEST(CpuKvOverRoceTransProviderTest, ExposesItsNameAndConfiguration)
{
    const TransportConfig config = MakeRoceConfig();
    roce::CpuKvOverRoceTransProvider provider(config);

    EXPECT_STREQ("CpuKvOverRoceTransProvider", provider.Name());
    EXPECT_EQ(config.nodeId, provider.Config().nodeId);
    EXPECT_EQ(2U, provider.Config().qpNum);
    ASSERT_EQ(1U, provider.Config().endpoints.size());
    EXPECT_EQ("10.0.0.11", provider.Config().endpoints.front().ip);
    EXPECT_EQ(19001, provider.Config().endpoints.front().port);
}

TEST(CpuKvOverRoceTransProviderTest, CreateConnectionFailsWhenThePeerIsUnreachable)
{
    roce::CpuKvOverRoceTransProvider provider(MakeRoceConfig());
    std::vector<TransProvider::ConnectionHandle> handles(3, FakeHandle(1));

    // Port 1 on loopback has no listener, so the out-of-band connect fails fast
    // and the provider has to report it without handing out any handle.
    const Status status = provider.CreateConnection("127.0.0.1", "127.0.0.1", 1, 2, 500, handles);

    EXPECT_FALSE(status.ok());
    EXPECT_NE(std::string::npos, status.message.find("out-of-band TCP connect"));
    EXPECT_TRUE(handles.empty());
    EXPECT_EQ(0U, provider.ConnectionCount());
}

TEST(CpuKvOverRoceTransProviderTest, CapabilityCallsRejectUnknownHandles)
{
    roce::CpuKvOverRoceTransProvider provider(MakeRoceConfig());

    // A handle that was never handed out is rejected before any capability is
    // reported, so a stale handle cannot be mistaken for a valid peer.
    ServerKvCapabilities capabilities{};
    capabilities.queueNum = 7;
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
              provider.GetServerCapabilities(nullptr, capabilities).code);
    EXPECT_EQ(0U, capabilities.queueNum);

    std::uint32_t tokenId = 99;
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, provider.GetMemTokenId(0, tokenId).code);
    EXPECT_EQ(0U, tokenId);
}

TEST(CpuKvOverRoceTransProviderTest, PerItemCallsMirrorTheInputSize)
{
    roce::CpuKvOverRoceTransProvider provider(MakeRoceConfig());

    const std::vector<TransProvider::ConnectionHandle> connections{FakeHandle(1), FakeHandle(2)};
    const std::vector<Status> deleted = provider.DeleteConnections(connections);
    ASSERT_EQ(connections.size(), deleted.size());
    // Unknown handles are reported per item instead of aborting the whole call.
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, deleted.front().code);

    const std::vector<TransProvider::UnregisterMemoryDesc> unregisterDescs{{1}, {2}, {3}};
    const std::vector<Status> unregistered = provider.UnregisterMemory(unregisterDescs);
    ASSERT_EQ(unregisterDescs.size(), unregistered.size());
    // Nothing was registered, so every handle is unknown rather than ignored.
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, unregistered.back().code);

    const std::vector<TransProvider::SendIoBatch> batches{
        {FakeHandle(1), nullptr, nullptr, 0}
    };
    const std::vector<Status> sent = provider.Send(batches, 1, 1);
    ASSERT_EQ(batches.size(), sent.size());
    EXPECT_EQ(StatusCode::UNSUPPORTED, sent.front().code);
}

TEST(CpuKvOverRoceTransProviderTest, ThreadHooksSucceedWithoutAllocatingAnything)
{
    roce::CpuKvOverRoceTransProvider provider(MakeRoceConfig());

    std::vector<TransProvider::ThreadHandle> threads{FakeHandle(1)};
    EXPECT_TRUE(provider.AllocThread(4, {1, 2, 3, 4}, threads).ok());
    EXPECT_TRUE(threads.empty());

    const std::vector<Status> freed = provider.FreeThread({FakeHandle(1)});
    ASSERT_EQ(1U, freed.size());
    EXPECT_TRUE(freed.front().ok());
}

TEST(CpuKvOverRoceTransProviderTest, FactoryReturnsTheProviderBehindTheInterface)
{
    const std::shared_ptr<TransProvider> provider =
        roce::CreateCpuKvOverRoceTransProvider(MakeRoceConfig());

    ASSERT_NE(nullptr, provider);
    EXPECT_NE(nullptr, dynamic_cast<roce::CpuKvOverRoceTransProvider*>(provider.get()));
    const auto* concrete = dynamic_cast<const roce::CpuKvOverRoceTransProvider*>(provider.get());
    ASSERT_NE(nullptr, concrete);
    EXPECT_EQ(MakeRoceConfig().nodeId, concrete->Config().nodeId);
    EXPECT_EQ(2U, concrete->Config().qpNum);
}

TEST(CpuKvOverRoceTransProviderTest, CreateTransProviderDispatchesOnCpuProviderType)
{
    TransportConfig config = MakeRoceConfig();
    config.providerType = TransProviderType::CPU;

    std::shared_ptr<TransProvider> provider;
    const Status status = CreateTransProvider(config, provider);

    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_NE(nullptr, provider);
    EXPECT_NE(nullptr, dynamic_cast<roce::CpuKvOverRoceTransProvider*>(provider.get()));
}

TEST(CpuKvOverRoceTransProviderTest, UnknownProviderTypeIsRejected)
{
    TransportConfig config = MakeRoceConfig();
    config.providerType = TransProviderType::UNSUPPORTED;

    std::shared_ptr<TransProvider> provider;
    const Status status = CreateTransProvider(config, provider);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(StatusCode::UNSUPPORTED, status.code);
    EXPECT_EQ(nullptr, provider);
}

TEST(CpuKvOverRoceTransProviderTest, CpuProviderCanBeSelectedFromConfiguration)
{
    EXPECT_EQ(TransProviderType::CPU, ParseConfigTransProviderType("CPU"));
    EXPECT_EQ(TransProviderType::CPU, ParseConfigTransProviderType("cpu"));
    EXPECT_EQ(TransProviderType::FAKE, ParseConfigTransProviderType("fake"));
    EXPECT_EQ(TransProviderType::UNSUPPORTED, ParseConfigTransProviderType("roce"));
}

}  // namespace
}  // namespace kv::test
