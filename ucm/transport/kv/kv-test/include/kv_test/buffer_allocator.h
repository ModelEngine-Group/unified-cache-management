#pragma once

#include "kv_test/kv_test_types.h"

namespace UC::KVTest {

class BufferAllocator {
public:
    Status BuildStoreBuffers(const GeneratedData& data, BufferSet& buffers) const;
    Status BuildRetrieveBuffers(const GeneratedData& data, BufferSet& buffers) const;
};

}  // namespace UC::KVTest
