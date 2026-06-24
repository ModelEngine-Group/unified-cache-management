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

std::int32_t ResolveFakeBackendDeviceId(const KvTestConfig& config)
{
    if (config.asuClientConfig.transportConfigs.empty()) { return kFakeBackendAclDeviceId; }

    const auto& transportConfig = config.asuClientConfig.transportConfigs.front();
    auto deviceIter = transportConfig.attrs.find("fake_backend.device_id");
    if (deviceIter != transportConfig.attrs.end() && !deviceIter->second.empty()) {
        return static_cast<std::int32_t>(std::stol(deviceIter->second));
    }
    if (!transportConfig.endpoints.empty()) { return transportConfig.endpoints.front().deviceId; }
    return kFakeBackendAclDeviceId;
}

Status SetUpAclThreadDevice(std::int32_t deviceId, bool* initialized)
{
    thread_local std::int32_t readyDeviceId = -1;
    if (readyDeviceId == deviceId) { return Status::Success(); }

    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
        return Status::Error(kExitInvalidArgument,
                             "fake_backend aclInit failed: ret=" + std::to_string(ret));
    }
    if (initialized != nullptr) { *initialized = ret == ACL_SUCCESS; }

    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        return Status::Error(kExitInvalidArgument,
                             "fake_backend aclrtSetDevice failed: device_id=" +
                                 std::to_string(deviceId) + " ret=" + std::to_string(ret));
    }
    readyDeviceId = deviceId;
    return Status::Success();
}

}  // namespace

FakeBackendAclRuntime::~FakeBackendAclRuntime() { TearDown(); }

Status FakeBackendAclRuntime::MaybeSetUp(const KvTestConfig& config)
{
    if (!IsFakeBackendMode(config)) { return Status::Success(); }

    deviceId_ = ResolveFakeBackendDeviceId(config);
    auto status = SetUpAclThreadDevice(deviceId_, &initialized_);
    if (!status.Ok()) {
        TearDown();
        return status;
    }
    deviceSet_ = true;
    return Status::Success();
}

void FakeBackendAclRuntime::TearDown()
{
    if (deviceSet_) {
        (void)aclrtResetDevice(deviceId_);
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

Status MaybeSetUpFakeBackendAclThread(const KvTestConfig& config)
{
    if (!IsFakeBackendMode(config)) { return Status::Success(); }
    return SetUpAclThreadDevice(ResolveFakeBackendDeviceId(config), nullptr);
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
