#pragma once

#include "kv_test/kv_test_types.h"

namespace UC::KVTest {

class ConsistencyChecker {
public:
    Status CheckStoreResult(const GeneratedData& expected, const BufferSet& retrieved,
                            const CommandResult& result, ConsistencySummary& summary) const;
    Status CheckRetrieveResult(const GeneratedData& expected, const BufferSet& retrieved,
                               const CommandResult& result, ConsistencySummary& summary) const;
    Status CheckDeleteResult(const std::vector<UC::ASU::CacheKey>& keys,
                             const CommandResult& deleteResult, const CommandResult& existResult,
                             ConsistencySummary& summary) const;
    Status CheckExistResult(const std::vector<UC::ASU::CacheKey>& keys, const CommandResult& result,
                            bool expectedExists, ConsistencySummary& summary) const;
};

}  // namespace UC::KVTest
