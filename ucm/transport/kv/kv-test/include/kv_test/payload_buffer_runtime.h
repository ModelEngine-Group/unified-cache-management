#pragma once

#include <cstdint>
#include "kv_test/kv_test_types.h"

namespace UC::KVTest {

class PayloadBufferAclRuntime {
public:
    PayloadBufferAclRuntime() = default;
    ~PayloadBufferAclRuntime();

    Status MaybeSetUp(const KvTestConfig& config);
    void TearDown();

private:
    bool initialized_{false};
    bool deviceSet_{false};
    std::int32_t deviceId_{0};
};

bool UsesDevicePayloadBuffers(const KvTestConfig& config);
Status MaybeSetUpPayloadAclThread(const KvTestConfig& config);

}  // namespace UC::KVTest
