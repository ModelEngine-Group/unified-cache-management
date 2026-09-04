#include "kv_test_config_helpers.h"
#include <algorithm>
#include <string>
#include <utility>

namespace UC::KVTest {
namespace {

constexpr int kFakeBackendAclDeviceId = 0;

bool HasProvider(const KvTestConfig& config, UC::ASU::TransProviderType providerType)
{
    return std::any_of(config.asuClientConfig.transportConfigs.begin(),
                       config.asuClientConfig.transportConfigs.end(),
                       [providerType](const UC::ASU::TransportConfig& transportConfig) {
                           return transportConfig.providerType == providerType;
                       });
}

void PatchFakeBackendTransportConfig(UC::ASU::TransportConfig& config,
                                     const KvTestFakeBackendConfig& fakeConfig)
{
    const auto fakeBackendDeviceId =
        config.deviceId < 0 ? kFakeBackendAclDeviceId : config.deviceId;
    config.deviceId = fakeBackendDeviceId;
    config.providerType = UC::ASU::TransProviderType::FAKE;
    config.attrs.try_emplace("kernel_count", "1");
    config.attrs.try_emplace("quiet_count", "1");
    config.attrs["kv_ns_id"] = std::to_string(config.asuId);
    config.attrs.try_emplace("dtype", "0");
    config.attrs.try_emplace("dspec", "0");
    config.attrs.try_emplace("lr", "false");
    config.attrs["fake_backend.path"] = fakeConfig.storePath;
    config.attrs["fake_backend.latency_ms"] = std::to_string(fakeConfig.latencyMs);
    config.attrs["fake_backend.worker_threads"] = std::to_string(fakeConfig.workerThreads);
    config.attrs["fake_backend.device_id"] = std::to_string(fakeBackendDeviceId);
    if (config.endpoints.empty()) {
        UC::ASU::AsuEndpoint endpoint;
        endpoint.ip = "fake_backend";
        endpoint.port = 19001;
        endpoint.protocol = UC::ASU::Protocol::TCP;
        config.endpoints.emplace_back(std::move(endpoint));
    }
}

}  // namespace

bool HasFakeProvider(const KvTestConfig& config)
{
    return HasProvider(config, UC::ASU::TransProviderType::FAKE);
}

bool IsAivProviderMode(const KvTestConfig& config)
{
    return HasProvider(config, UC::ASU::TransProviderType::AIV);
}

DeviceAllocationPolicy AllocationPolicyForConfig(const KvTestConfig& config)
{
    return IsAivProviderMode(config) ? DeviceAllocationPolicy::AIV_REGISTERABLE
                                     : DeviceAllocationPolicy::DEFAULT;
}

void MaybePrepareFakeBackend(KvTestConfig& config)
{
    if (!HasFakeProvider(config)) { return; }

    if (config.fakeBackend.storePath.empty()) {
        config.fakeBackend.storePath = "./kv-test-fake-backend-store";
    }

    config.asuClientConfig.attrs.try_emplace("hash_table.type", "RING_HASH");
    config.asuClientConfig.attrs.try_emplace("ring_hash.virtual_node_count", "128");
    for (auto& transportConfig : config.asuClientConfig.transportConfigs) {
        if (transportConfig.providerType == UC::ASU::TransProviderType::FAKE) {
            PatchFakeBackendTransportConfig(transportConfig, config.fakeBackend);
        }
    }
}

}  // namespace UC::KVTest
