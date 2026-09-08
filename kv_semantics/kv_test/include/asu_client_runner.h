#pragma once

#include <memory>
#include "kv_test_types.h"

namespace kv::bench {

kv::TaskResult BuildEmptyTaskResult();

class KvClientRunner {
public:
    explicit KvClientRunner(std::unique_ptr<kv::KvClient> client);
    ~KvClientRunner();

    Status Init(const KvTestConfig& config);
    Status Shutdown();

    Status RegisterBuffers(BufferSet& buffers);
    Status UnregisterBuffers(const BufferSet& buffers);

    // SINGLE_ENTRY_PER_CALL maps single Store/Retrieve to one KvClient call per entry.
    // ALL_ENTRIES_IN_ONE_CALL maps batch commands to one KvClient call with all entries.
    Status Store(const BufferSet& buffers, SubmitMode submitMode, std::uint64_t timeoutMs,
                 CommandResult& result);
    Status Retrieve(const BufferSet& buffers, SubmitMode submitMode, std::uint64_t timeoutMs,
                    CommandResult& result);
    Status SubmitStore(const BufferSet& buffers, SubmitMode submitMode, kv::TaskId& taskId);
    Status SubmitRetrieve(const BufferSet& buffers, SubmitMode submitMode, kv::TaskId& taskId);
    Status Wait(kv::TaskId taskId, std::uint64_t timeoutMs, CommandResult& result);
    Status Delete(const std::vector<kv::CacheKey>& keys, std::uint64_t timeoutMs,
                  CommandResult& result);
    Status Exist(const std::vector<kv::CacheKey>& keys, std::uint64_t timeoutMs,
                 CommandResult& result);

private:
    std::unique_ptr<kv::KvClient> client_;
};

}  // namespace kv::bench
