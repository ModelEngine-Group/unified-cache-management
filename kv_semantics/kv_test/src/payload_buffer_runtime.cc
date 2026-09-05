#include "payload_buffer_runtime.h"
#include <algorithm>
#include <cstdlib>
#include "kv_test_config_helpers.h"

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
    if (transportConfig.deviceId >= 0) { return transportConfig.deviceId; }
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
        if (transportConfig.deviceId >= 0) { return transportConfig.deviceId; }
    }
    return kDefaultPayloadAclDeviceId;
}

namespace {

Status SetUpThreadDevice(Trans::Device& device, std::int32_t deviceId, bool* initialized)
{
    thread_local std::int32_t readyDeviceId = -1;
    if (readyDeviceId == deviceId) { return Status::Success(); }

    const auto initStatus = device.Init();
    if (!initStatus.ok()) {
        return Status::Error(kExitInvalidArgument,
                             "payload buffer Device::Init failed: " + initStatus.message);
    }
    if (initialized != nullptr) { *initialized = initStatus.ok(); }

    const auto setupStatus = device.Setup(deviceId);
    if (!setupStatus.ok()) {
        return Status::Error(kExitInvalidArgument,
                             "payload buffer Device::Setup failed: device_id=" +
                                 std::to_string(deviceId) + " " + setupStatus.message);
    }
    readyDeviceId = deviceId;
    return Status::Success();
}

}  // namespace

PayloadBufferRuntime::~PayloadBufferRuntime() { TearDown(); }

Status PayloadBufferRuntime::MaybeSetUp(const KvTestConfig& config)
{
    if (!UsesDevicePayloadBuffers(config)) { return Status::Success(); }

    deviceId_ = ResolvePayloadDeviceId(config);
    auto status = SetUpThreadDevice(device_, deviceId_, &initialized_);
    if (!status.Ok()) {
        TearDown();
        return status;
    }
    deviceSet_ = true;
    return Status::Success();
}

void PayloadBufferRuntime::TearDown()
{
    if (deviceSet_) {
        (void)device_.Reset(deviceId_);
        deviceSet_ = false;
    }
    if (initialized_) {
        (void)device_.Finalize();
        initialized_ = false;
    }
}

bool UsesDevicePayloadBuffers(const KvTestConfig& config)
{
    return HasFakeProvider(config) || IsAivProviderMode(config);
}

Status MaybeSetUpPayloadThread(const KvTestConfig& config)
{
    if (!UsesDevicePayloadBuffers(config)) { return Status::Success(); }
    Trans::Device device;
    return SetUpThreadDevice(device, ResolvePayloadDeviceId(config), nullptr);
}

}  // namespace UC::KVTest
