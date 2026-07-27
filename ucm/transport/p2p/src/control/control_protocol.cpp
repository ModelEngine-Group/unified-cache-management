#include "control/control_protocol.h"
#include "common/binary_codec.h"

namespace transport {

Status EncodeControlRequest(const ControlRequest& request, Metadata& out)
{
    out.clear();
    if (!detail::AppendU32(out, static_cast<uint32_t>(request.operation))) {
        return Status::InvalidParam();
    }
    if (request.operation == ControlOperation::ExchangeMetadata) {
        return detail::AppendString(out, request.manager_id) &&
                       detail::AppendBytes(out, request.payload)
                   ? Status::OK()
                   : Status::InvalidParam();
    }
    if (!request.protocol.has_value()) { return Status::InvalidParam(); }
    return detail::AppendU32(out, static_cast<uint32_t>(*request.protocol)) &&
                   detail::AppendString(out, request.manager_id)
               ? Status::OK()
               : Status::InvalidParam();
}

Status DecodeControlRequest(const Metadata& in, ControlRequest& request)
{
    size_t offset = 0;
    uint32_t raw_operation = 0;
    if (!detail::ReadU32(in, offset, raw_operation) ||
        raw_operation > static_cast<uint32_t>(ControlOperation::Disconnect)) {
        return Status::InvalidParam();
    }
    request.operation = static_cast<ControlOperation>(raw_operation);
    if (request.operation == ControlOperation::ExchangeMetadata) {
        return detail::ReadString(in, offset, request.manager_id) &&
                       detail::ReadBytes(in, offset, request.payload) && offset == in.size()
                   ? Status::OK()
                   : Status::InvalidParam();
    }

    uint32_t raw_protocol = 0;
    if (!detail::ReadU32(in, offset, raw_protocol) ||
        !detail::ReadString(in, offset, request.manager_id) || offset != in.size()) {
        return Status::InvalidParam();
    }
    request.protocol = static_cast<TransportProtocol>(raw_protocol);
    return Status::OK();
}

}  // namespace transport
