#pragma once

#include "kv_test/kv_test_types.h"

namespace UC::KVTest {

bool IsFakeBackendMode(const KvTestConfig& config);
bool IsAivProviderMode(const KvTestConfig& config);
void MaybePrepareFakeBackend(KvTestConfig& config);

}  // namespace UC::KVTest
