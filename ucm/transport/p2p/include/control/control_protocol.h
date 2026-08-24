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

inline const char* ControlOperationName(ControlOperation operation) noexcept
{
    switch (operation) {
        case ControlOperation::ExchangeMetadata: return "exchange-metadata";
        case ControlOperation::Connect: return "connect";
        case ControlOperation::Disconnect: return "disconnect";
    }
    return "unknown";
}

struct ControlRequest {
    ControlOperation operation;
    std::optional<TransportProtocol> protocol;
    ManagerID manager_id;
    Metadata payload;
};

Status EncodeControlRequest(const ControlRequest& request, Metadata& out);
Status DecodeControlRequest(const Metadata& in, ControlRequest& request);

}  // namespace transport
