#include "kv_test/fake_backend.h"
#include <acl/acl.h>
#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

namespace UC::KVTest {
namespace {

constexpr int kExitInvalidArgument = 1;
constexpr int kFakeBackendAclDeviceId = 0;

std::string NormalizeMode(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

void PatchTransportConfig(UC::ASU::TransportConfig& config,
                          const KvTestFakeBackendConfig& fakeConfig)
{
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
    config.attrs["fake_backend.device_id"] = std::to_string(kFakeBackendAclDeviceId);
    if (config.endpoints.empty()) {
        UC::ASU::AsuEndpoint endpoint;
        endpoint.ip = "fake_backend";
        endpoint.port = 19001;
        endpoint.protocol = UC::ASU::Protocol::TCP;
        endpoint.deviceId = kFakeBackendAclDeviceId;
        config.endpoints.emplace_back(std::move(endpoint));
    }
}

}  // namespace

FakeBackendAclRuntime::~FakeBackendAclRuntime() { TearDown(); }

Status FakeBackendAclRuntime::MaybeSetUp(const KvTestConfig& config)
{
    if (!IsFakeBackendMode(config)) { return Status::Success(); }

    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        return Status::Error(kExitInvalidArgument,
                             "fake_backend aclInit failed: " + std::to_string(ret));
    }
    initialized_ = true;

    // kv-test fake_backend is a temporary standalone test path. Use device 0 until the
    // ASU client/transport runtime contract is formalized.
    ret = aclrtSetDevice(kFakeBackendAclDeviceId);
    if (ret != ACL_SUCCESS) {
        return Status::Error(kExitInvalidArgument,
                             "fake_backend aclrtSetDevice failed: device_id=" +
                                 std::to_string(kFakeBackendAclDeviceId) +
                                 " ret=" + std::to_string(ret));
    }
    deviceSet_ = true;
    return Status::Success();
}

void FakeBackendAclRuntime::TearDown()
{
    if (deviceSet_) {
        (void)aclrtResetDevice(kFakeBackendAclDeviceId);
        deviceSet_ = false;
    }
    if (initialized_) {
        (void)aclFinalize();
        initialized_ = false;
    }
}

bool IsFakeBackendMode(const KvTestConfig& config)
{
    const auto mode = NormalizeMode(config.asuClientMode);
    return mode == "fake_backend" || mode == "fakebackend";
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
        PatchTransportConfig(transportConfig, config.fakeBackend);
    }
}

}  // namespace UC::KVTest
