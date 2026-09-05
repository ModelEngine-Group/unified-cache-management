#pragma once

#include "kv_test_types.h"

namespace kv::bench {

class KvTestConfigLoader {
public:
    Status ResolveConfigPath(const std::string& configPath, std::string& resolvedPath) const;

    // Loads the existing AsuClientConfig key-value format and derives kv-test fields.
    Status Load(const std::string& configPath, KvTestConfig& config) const;
    Status MergeCommandOptions(const CommandOptions& options, KvTestConfig& config) const;
};

}  // namespace kv::bench
