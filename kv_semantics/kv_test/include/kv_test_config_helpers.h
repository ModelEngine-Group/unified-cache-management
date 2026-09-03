#pragma once

#include "kv_test_types.h"

namespace UC::KVTest {

bool HasFakeProvider(const KvTestConfig& config);
bool IsAivProviderMode(const KvTestConfig& config);
DeviceAllocationPolicy AllocationPolicyForConfig(const KvTestConfig& config);
void MaybePrepareFakeBackend(KvTestConfig& config);

}  // namespace UC::KVTest
