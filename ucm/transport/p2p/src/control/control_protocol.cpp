/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
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
