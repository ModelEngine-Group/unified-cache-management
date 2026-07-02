#pragma once

#include "kv_test/asu_client_runner.h"
#include "kv_test/kv_test_types.h"

namespace UC::KVTest {

std::string FormatMiBPerSec(double bytesPerSec);

class BenchRunner {
public:
    Status Run(const CommandOptions& options, const KvTestConfig& config,
               AsuClientRunner& clientRunner, CommandResult& result) const;
};

}  // namespace UC::KVTest
