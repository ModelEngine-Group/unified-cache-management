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
#include "core/transport.h"

namespace transport {

enum class ManagerMessageType : uint8_t {
    ConnectRequest = 0,
    DisconnectRequest = 1,
    ControlResponse = 2,
    Data = 3,
};

#pragma pack(push, 1)
struct ManagerMessageProtocol {
    uint32_t body_size;
    // Reserved for Data messages.
    uint64_t request_id;
    // Reserved for ControlResponse and Data messages.
    TransportProtocol protocol;
    // Reserved for request and Data messages.
    int32_t status;
    uint32_t source_size;
    uint32_t payload_size;
    ManagerMessageType type;
    uint8_t reserved[7];
};
#pragma pack(pop)

static_assert(sizeof(ManagerMessageProtocol) == 36, "unexpected manager message protocol size");

}  // namespace transport
