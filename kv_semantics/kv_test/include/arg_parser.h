#pragma once

#include "kv_test_types.h"

namespace UC::KVTest {

std::string CommandTypeName(CommandType command);
std::string BenchOpTypeName(BenchOpType op);

class ArgParser {
public:
    Status Parse(int argc, char** argv, CommandOptions& options) const;
};

}  // namespace UC::KVTest
