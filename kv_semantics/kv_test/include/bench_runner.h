#pragma once

#include "asu_client_runner.h"
#include "kv_test_types.h"

namespace UC::KVTest {

std::string FormatMiBPerSec(double bytesPerSec);

class BenchRunner {
public:
    Status Run(const CommandOptions& options, const KvTestConfig& config,
               AsuClientRunner& clientRunner, CommandResult& result) const;
};

}  // namespace UC::KVTest
