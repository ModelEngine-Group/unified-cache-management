#pragma once

#include "kv_test/kv_test_types.h"

namespace UC::KVTest {

class ArgParser {
public:
    Status Parse(int argc, char** argv, CommandOptions& options) const;
};

}  // namespace UC::KVTest
