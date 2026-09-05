#pragma once

#include "kv_test_types.h"

namespace kv::bench {

Status StringToCacheKey(const std::string& value, const std::string& source,
                        kv::CacheKey& key);
Status ValidateGeneratedData(const GeneratedData& data, const std::string& operation);

class KeyValueGenerator {
public:
    // Generates deterministic value bytes from key, seed, and value-size.
    Status Generate(const CommandOptions& options, const KvTestConfig& config,
                    GeneratedData& data) const;
    // Uses CRC64 by default; digest is for consistency logs and result files.
    Status Digest(const std::vector<std::uint8_t>& value, std::string& digest) const;
};

}  // namespace kv::bench
