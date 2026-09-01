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
