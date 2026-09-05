#pragma once

#include "kv_test_types.h"

namespace kv::bench {

std::string CommandTypeName(CommandType command);
std::string BenchOpTypeName(BenchOpType op);

class ArgParser {
public:
    Status Parse(int argc, char** argv, CommandOptions& options) const;
};

}  // namespace kv::bench
