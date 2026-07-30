#pragma once

#include <cstdint>
#include <optional>
#include "core/transport.h"

namespace transport {

enum class ControlOperation : uint32_t {
    ExchangeMetadata = 0,
    Connect = 1,
    Disconnect = 2,
};

struct ControlRequest {
    ControlOperation operation;
    std::optional<TransportProtocol> protocol;
    ManagerID manager_id;
    Metadata payload;
};

Status EncodeControlRequest(const ControlRequest& request, Metadata& out);
Status DecodeControlRequest(const Metadata& in, ControlRequest& request);

}  // namespace transport
