#include "kv_test_config_helpers.h"
#include <gtest/gtest.h>
#include <utility>

namespace UC::KVTest {
namespace {

TEST(KvTestConfigHelpersTest, AivProviderDoesNotEnableFakeBackend)
{
    KvTestConfig config;
    UC::ASU::TransportConfig transportConfig;
    transportConfig.providerType = UC::ASU::TransProviderType::AIV;
    config.asuClientConfig.transportConfigs.emplace_back(std::move(transportConfig));

    EXPECT_FALSE(HasFakeProvider(config));
    MaybePrepareFakeBackend(config);
    EXPECT_EQ(config.asuClientConfig.transportConfigs.front().providerType,
              UC::ASU::TransProviderType::AIV);
    EXPECT_TRUE(config.asuClientConfig.transportConfigs.front().attrs.empty());
    EXPECT_EQ(AllocationPolicyForConfig(config), DeviceAllocationPolicy::AIV_REGISTERABLE);
}

TEST(KvTestConfigHelpersTest, FakeDefaultsDoNotModifyAivTransport)
{
    KvTestConfig config;
    UC::ASU::TransportConfig fakeConfig;
    fakeConfig.asuId = 1;
    fakeConfig.providerType = UC::ASU::TransProviderType::FAKE;
    fakeConfig.attrs["sc"] = "false";
    config.asuClientConfig.transportConfigs.emplace_back(std::move(fakeConfig));

    UC::ASU::TransportConfig aivConfig;
    aivConfig.asuId = 2;
    aivConfig.providerType = UC::ASU::TransProviderType::AIV;
    aivConfig.attrs["sentinel"] = "unchanged";
    config.asuClientConfig.transportConfigs.emplace_back(std::move(aivConfig));

    ASSERT_TRUE(HasFakeProvider(config));
    MaybePrepareFakeBackend(config);

    const auto& patchedFake = config.asuClientConfig.transportConfigs[0];
    EXPECT_EQ(patchedFake.providerType, UC::ASU::TransProviderType::FAKE);
    EXPECT_EQ(patchedFake.attrs.at("sc"), "false");
    EXPECT_EQ(patchedFake.attrs.at("fake_backend.path"), "./kv-test-fake-backend-store");
    EXPECT_EQ(patchedFake.attrs.at("fake_backend.worker_threads"), "4");

    const auto& unchangedAiv = config.asuClientConfig.transportConfigs[1];
    EXPECT_EQ(unchangedAiv.providerType, UC::ASU::TransProviderType::AIV);
    EXPECT_EQ(unchangedAiv.attrs.size(), 1);
    EXPECT_EQ(unchangedAiv.attrs.at("sentinel"), "unchanged");
    EXPECT_EQ(AllocationPolicyForConfig(config), DeviceAllocationPolicy::AIV_REGISTERABLE);
}

TEST(KvTestConfigHelpersTest, FakeProviderUsesDefaultDeviceAllocation)
{
    KvTestConfig config;
    UC::ASU::TransportConfig transportConfig;
    transportConfig.providerType = UC::ASU::TransProviderType::FAKE;
    config.asuClientConfig.transportConfigs.emplace_back(std::move(transportConfig));

    EXPECT_EQ(AllocationPolicyForConfig(config), DeviceAllocationPolicy::DEFAULT);
}

}  // namespace
}  // namespace UC::KVTest
