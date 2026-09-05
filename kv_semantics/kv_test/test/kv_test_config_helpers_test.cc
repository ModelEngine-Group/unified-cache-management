#include "kv_test_config_helpers.h"
#include <gtest/gtest.h>
#include <utility>

namespace kv::bench {
namespace {

TEST(KvTestConfigHelpersTest, AivProviderDoesNotEnableFakeBackend)
{
    KvTestConfig config;
    kv::TransportConfig transportConfig;
    transportConfig.providerType = kv::TransProviderType::AIV;
    config.asuClientConfig.transportConfigs.emplace_back(std::move(transportConfig));

    EXPECT_FALSE(HasFakeProvider(config));
    MaybePrepareFakeBackend(config);
    EXPECT_EQ(config.asuClientConfig.transportConfigs.front().providerType,
              kv::TransProviderType::AIV);
    EXPECT_TRUE(config.asuClientConfig.transportConfigs.front().attrs.empty());
    EXPECT_EQ(AllocationPolicyForConfig(config), DeviceAllocationPolicy::AIV_REGISTERABLE);
}

TEST(KvTestConfigHelpersTest, FakeDefaultsDoNotModifyAivTransport)
{
    KvTestConfig config;
    kv::TransportConfig fakeConfig;
    fakeConfig.asuId = 1;
    fakeConfig.providerType = kv::TransProviderType::FAKE;
    fakeConfig.attrs["sc"] = "false";
    config.asuClientConfig.transportConfigs.emplace_back(std::move(fakeConfig));

    kv::TransportConfig aivConfig;
    aivConfig.asuId = 2;
    aivConfig.providerType = kv::TransProviderType::AIV;
    aivConfig.attrs["sentinel"] = "unchanged";
    config.asuClientConfig.transportConfigs.emplace_back(std::move(aivConfig));

    ASSERT_TRUE(HasFakeProvider(config));
    MaybePrepareFakeBackend(config);

    const auto& patchedFake = config.asuClientConfig.transportConfigs[0];
    EXPECT_EQ(patchedFake.providerType, kv::TransProviderType::FAKE);
    EXPECT_EQ(patchedFake.attrs.at("sc"), "false");
    EXPECT_EQ(patchedFake.attrs.at("fake_backend.path"), "./kv-test-fake-backend-store");
    EXPECT_EQ(patchedFake.attrs.at("fake_backend.worker_threads"), "4");

    const auto& unchangedAiv = config.asuClientConfig.transportConfigs[1];
    EXPECT_EQ(unchangedAiv.providerType, kv::TransProviderType::AIV);
    EXPECT_EQ(unchangedAiv.attrs.size(), 1);
    EXPECT_EQ(unchangedAiv.attrs.at("sentinel"), "unchanged");
    EXPECT_EQ(AllocationPolicyForConfig(config), DeviceAllocationPolicy::AIV_REGISTERABLE);
}

TEST(KvTestConfigHelpersTest, FakeProviderUsesDefaultDeviceAllocation)
{
    KvTestConfig config;
    kv::TransportConfig transportConfig;
    transportConfig.providerType = kv::TransProviderType::FAKE;
    config.asuClientConfig.transportConfigs.emplace_back(std::move(transportConfig));

    EXPECT_EQ(AllocationPolicyForConfig(config), DeviceAllocationPolicy::DEFAULT);
}

}  // namespace
}  // namespace kv::bench
