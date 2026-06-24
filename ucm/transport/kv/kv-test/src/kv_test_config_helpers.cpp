#include "kv_test/kv_test_config_helpers.h"
#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

namespace UC::KVTest {
namespace {

constexpr int kFakeBackendAclDeviceId = 0;

std::string NormalizeMode(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

void PatchFakeBackendTransportConfig(UC::ASU::TransportConfig& config,
                                     const KvTestFakeBackendConfig& fakeConfig)
{
    const auto fakeBackendDeviceId =
        config.endpoints.empty() ? kFakeBackendAclDeviceId : config.endpoints.front().deviceId;
    config.providerType = UC::ASU::TransProviderType::FAKE;
    config.attrs.try_emplace("kernel_count", "1");
    config.attrs.try_emplace("quiet_count", "1");
    config.attrs["kv_ns_id"] = std::to_string(config.asuId);
    config.attrs.try_emplace("dtype", "0");
    config.attrs.try_emplace("dspec", "0");
    config.attrs.try_emplace("lr", "false");
    config.attrs["sc"] = "true";
    config.attrs["fake_backend.path"] = fakeConfig.storePath;
    config.attrs["fake_backend.latency_ms"] = std::to_string(fakeConfig.latencyMs);
    config.attrs["fake_backend.device_id"] = std::to_string(fakeBackendDeviceId);
    if (config.endpoints.empty()) {
        UC::ASU::AsuEndpoint endpoint;
        endpoint.ip = "fake_backend";
        endpoint.port = 19001;
        endpoint.protocol = UC::ASU::Protocol::TCP;
        endpoint.deviceId = fakeBackendDeviceId;
        config.endpoints.emplace_back(std::move(endpoint));
    }
}

}  // namespace

bool IsFakeBackendMode(const KvTestConfig& config)
{
    const auto mode = NormalizeMode(config.asuClientMode);
    return mode == "fake_backend" || mode == "fakebackend";
}

bool IsAivProviderMode(const KvTestConfig& config)
{
    return std::any_of(config.asuClientConfig.transportConfigs.begin(),
                       config.asuClientConfig.transportConfigs.end(),
                       [](const UC::ASU::TransportConfig& transportConfig) {
                           return transportConfig.providerType == UC::ASU::TransProviderType::AIV;
                       });
}

void MaybePrepareFakeBackend(KvTestConfig& config)
{
    if (!IsFakeBackendMode(config)) { return; }

    if (config.fakeBackend.storePath.empty()) {
        config.fakeBackend.storePath =
            config.localStorePath.empty() ? "./kv-test-fake-backend-store" : config.localStorePath;
    }

    config.asuClientConfig.attrs.try_emplace("hash_table.type", "RING_HASH");
    config.asuClientConfig.attrs.try_emplace("ring_hash.virtual_node_count", "128");
    if (config.asuClientConfig.transportConfigs.empty()) {
        UC::ASU::TransportConfig transportConfig;
        transportConfig.asuId = 1;
        config.asuClientConfig.transportConfigs.emplace_back(std::move(transportConfig));
    }
    for (auto& transportConfig : config.asuClientConfig.transportConfigs) {
        PatchFakeBackendTransportConfig(transportConfig, config.fakeBackend);
    }
}

}  // namespace UC::KVTest
