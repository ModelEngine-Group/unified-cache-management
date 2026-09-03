#pragma once

#include <cstdint>
#include "kv_test_types.h"
#include "trans/device.h"

namespace UC::KVTest {

class PayloadBufferRuntime {
public:
    PayloadBufferRuntime() = default;
    ~PayloadBufferRuntime();

    Status MaybeSetUp(const KvTestConfig& config);
    void TearDown();

private:
    bool initialized_{false};
    bool deviceSet_{false};
    std::int32_t deviceId_{0};
    Trans::Device device_;
};

std::int32_t ResolvePayloadDeviceId(const KvTestConfig& config);
bool UsesDevicePayloadBuffers(const KvTestConfig& config);
Status MaybeSetUpPayloadThread(const KvTestConfig& config);

}  // namespace UC::KVTest
