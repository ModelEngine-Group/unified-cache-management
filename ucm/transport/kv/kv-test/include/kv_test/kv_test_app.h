#pragma once

#include "kv_test/arg_parser.h"
#include "kv_test/asu_client_runner.h"
#include "kv_test/bench_runner.h"
#include "kv_test/buffer_allocator.h"
#include "kv_test/consistency_checker.h"
#include "kv_test/hcomm_config_adapter.h"
#include "kv_test/key_value_generator.h"
#include "kv_test/kv_test_config_loader.h"
#include "kv_test/result_writer.h"

namespace UC::KVTest {

class KvTestApp {
public:
    KvTestApp();

    int Run(int argc, char** argv);

private:
    Status RunCommand(const CommandOptions& options, const KvTestConfig& config,
                      AsuClientRunner& clientRunner, CommandResult& result);
    Status RunStoreLikeCommand(const CommandOptions& options, const KvTestConfig& config,
                               AsuClientRunner& clientRunner, CommandResult& result);
    Status RunRetrieveLikeCommand(const CommandOptions& options, const KvTestConfig& config,
                                  AsuClientRunner& clientRunner, CommandResult& result);
    Status RunDeleteCommand(const CommandOptions& options, const KvTestConfig& config,
                            AsuClientRunner& clientRunner, CommandResult& result);
    Status RunExistCommand(const CommandOptions& options, const KvTestConfig& config,
                           AsuClientRunner& clientRunner, CommandResult& result);

    ArgParser argParser_;
    BenchRunner benchRunner_;
    BufferAllocator bufferAllocator_;
    ConsistencyChecker consistencyChecker_;
    HcommConfigAdapter hcommConfigAdapter_;
    KeyValueGenerator generator_;
    KvTestConfigLoader configLoader_;
    ResultWriter resultWriter_;
};

}  // namespace UC::KVTest
