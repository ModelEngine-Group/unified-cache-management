#pragma once

#include "asu_client_runner.h"
#include "kv_test_types.h"

namespace kv::bench {

std::string FormatMiBPerSec(double bytesPerSec);

class BenchRunner {
public:
    Status Run(const CommandOptions& options, const KvTestConfig& config,
               AsuClientRunner& clientRunner, CommandResult& result) const;
};

}  // namespace kv::bench
