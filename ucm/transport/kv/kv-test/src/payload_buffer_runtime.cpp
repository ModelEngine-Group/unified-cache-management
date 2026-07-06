#include "kv_test/payload_buffer_runtime.h"
#include <acl/acl.h>
#include <algorithm>
#include <cstdlib>
#include "kv_test/kv_test_config_helpers.h"

namespace UC::KVTest {
namespace {

constexpr int kExitInvalidArgument = 1;
constexpr int kDefaultPayloadAclDeviceId = 0;

std::int32_t ResolveFakeBackendPayloadDeviceId(const KvTestConfig& config)
{
    if (config.asuClientConfig.transportConfigs.empty()) { return kDefaultPayloadAclDeviceId; }

    auto transportIter =
        std::find_if(config.asuClientConfig.transportConfigs.begin(),
                     config.asuClientConfig.transportConfigs.end(),
                     [](const UC::ASU::TransportConfig& transportConfig) {
                         return transportConfig.providerType == UC::ASU::TransProviderType::FAKE;
                     });
    if (transportIter == config.asuClientConfig.transportConfigs.end()) {
        transportIter = config.asuClientConfig.transportConfigs.begin();
    }
    const auto& transportConfig = *transportIter;
    auto deviceIter = transportConfig.attrs.find("fake_backend.device_id");
    if (deviceIter != transportConfig.attrs.end() && !deviceIter->second.empty()) {
        return static_cast<std::int32_t>(std::stol(deviceIter->second));
    }
    if (!transportConfig.endpoints.empty()) { return transportConfig.endpoints.front().deviceId; }
    return kDefaultPayloadAclDeviceId;
}

}  // namespace

std::int32_t ResolvePayloadDeviceId(const KvTestConfig& config)
{
    if (HasFakeProvider(config) && !IsAivProviderMode(config)) {
        return ResolveFakeBackendPayloadDeviceId(config);
    }

    if (const char* deviceId = std::getenv("UMC_ASU_DEVICE_ID");
        deviceId != nullptr && *deviceId != '\0') {
        return static_cast<std::int32_t>(std::stol(deviceId));
    }

    for (const auto& transportConfig : config.asuClientConfig.transportConfigs) {
        for (const auto& endpoint : transportConfig.endpoints) {
            if (endpoint.deviceId >= 0) { return endpoint.deviceId; }
        }
    }
    return kDefaultPayloadAclDeviceId;
}

namespace {

Status SetUpAclThreadDevice(std::int32_t deviceId, bool* initialized)
{
    thread_local std::int32_t readyDeviceId = -1;
    if (readyDeviceId == deviceId) { return Status::Success(); }

    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
        return Status::Error(kExitInvalidArgument,
                             "payload buffer aclInit failed: ret=" + std::to_string(ret));
    }
    if (initialized != nullptr) { *initialized = ret == ACL_SUCCESS; }

    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        return Status::Error(kExitInvalidArgument,
                             "payload buffer aclrtSetDevice failed: device_id=" +
                                 std::to_string(deviceId) + " ret=" + std::to_string(ret));
    }
    readyDeviceId = deviceId;
    return Status::Success();
}

}  // namespace

PayloadBufferAclRuntime::~PayloadBufferAclRuntime() { TearDown(); }

Status PayloadBufferAclRuntime::MaybeSetUp(const KvTestConfig& config)
{
    if (!UsesDevicePayloadBuffers(config)) { return Status::Success(); }

    deviceId_ = ResolvePayloadDeviceId(config);
    auto status = SetUpAclThreadDevice(deviceId_, &initialized_);
    if (!status.Ok()) {
        TearDown();
        return status;
    }
    deviceSet_ = true;
    return Status::Success();
}

void PayloadBufferAclRuntime::TearDown()
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

bool UsesDevicePayloadBuffers(const KvTestConfig& config)
{
    return HasFakeProvider(config) || IsAivProviderMode(config);
}

Status MaybeSetUpPayloadAclThread(const KvTestConfig& config)
{
    if (!UsesDevicePayloadBuffers(config)) { return Status::Success(); }
    return SetUpAclThreadDevice(ResolvePayloadDeviceId(config), nullptr);
}

}  // namespace UC::KVTest
