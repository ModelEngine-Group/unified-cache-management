#pragma once

#include "kv_test_types.h"

namespace kv::bench {

class HcommConfigAdapter {
public:
    // Maps kv::Protocol to the Hcomm protocol selected in questions.md.
    // TODO: Confirm UBOE and TCP->PCIE mapping with the final Hcomm header.
    Status ResolveProtocol(kv::Protocol protocol, const KvTestConfig& config,
                           HcommProtocol& hcommProtocol) const;

    // Hcomm local role is fixed to server; socket and port are read from ASU config.
    Status ValidateChannelSource(const KvTestConfig& config) const;
};

}  // namespace kv::bench
