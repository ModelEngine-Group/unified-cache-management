#pragma once

#include <memory>
#include "kv_test_types.h"

namespace kv::bench {

kv::TaskResult BuildEmptyTaskResult();

class AsuClientRunner {
public:
    explicit AsuClientRunner(std::unique_ptr<kv::AsuClient> client);
    ~AsuClientRunner();

    Status Init(const KvTestConfig& config);
    Status Shutdown();

    Status RegisterBuffers(BufferSet& buffers);
    Status UnregisterBuffers(const BufferSet& buffers);

    // SINGLE_ENTRY_PER_CALL maps single Store/Retrieve to one AsuClient call per entry.
    // ALL_ENTRIES_IN_ONE_CALL maps batch commands to one AsuClient call with all entries.
    Status Store(const BufferSet& buffers, SubmitMode submitMode, std::uint64_t timeoutMs,
                 CommandResult& result);
    Status Retrieve(const BufferSet& buffers, SubmitMode submitMode, std::uint64_t timeoutMs,
                    CommandResult& result);
    Status Delete(const std::vector<kv::CacheKey>& keys, std::uint64_t timeoutMs,
                  CommandResult& result);
    Status Exist(const std::vector<kv::CacheKey>& keys, std::uint64_t timeoutMs,
                 CommandResult& result);

private:
    std::unique_ptr<kv::AsuClient> client_;
};

}  // namespace kv::bench
