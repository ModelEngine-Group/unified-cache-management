#pragma once

#include "kv_test/kv_test_types.h"

namespace UC::KVTest {

class FakeBackendAclRuntime {
public:
    FakeBackendAclRuntime() = default;
    ~FakeBackendAclRuntime();

    Status MaybeSetUp(const KvTestConfig& config);
    void TearDown();

private:
    bool initialized_{false};
    bool deviceSet_{false};
};

bool IsFakeBackendMode(const KvTestConfig& config);
void MaybePrepareFakeBackend(KvTestConfig& config);

}  // namespace UC::KVTest
