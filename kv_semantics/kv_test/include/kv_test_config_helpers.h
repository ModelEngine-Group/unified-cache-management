#pragma once

#include "kv_test_types.h"

namespace kv::bench {

bool HasFakeProvider(const KvTestConfig& config);
bool IsAivProviderMode(const KvTestConfig& config);
DeviceAllocationPolicy AllocationPolicyForConfig(const KvTestConfig& config);
void MaybePrepareFakeBackend(KvTestConfig& config);

}  // namespace kv::bench
