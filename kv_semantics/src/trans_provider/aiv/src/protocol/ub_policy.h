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

#include <array>
#include <cstdint>

namespace umc::comm {

enum class OobTransportKind : int32_t {
    StaticConfig = 0,
    RaSocket = 1,
    Tcp = 2,
};

enum class TransportProfile : int32_t {
    Ubc = 0,
    Uboe = 1,
};

enum class JettyConnMode : int32_t {
    Rc = 0,
    Rm = 1,
};

enum class UbTpType : uint32_t {
    Rtp = 0,
    Ctp = 1,
    Utp = 2,
};

struct UbLocalNetAddr {
    TransportProfile kind{TransportProfile::Ubc};
    uint8_t prefixLen{0};  // IPv4: 0..32; IPv6: 0..128
    uint16_t family{0};    // AF_INET = 2 / AF_INET6 = 10 / 0
    uint64_t vlan{0};
    std::array<uint8_t, 6> mac{};
    std::array<uint8_t, 16> ipRaw{};  // IPv4-mapped IPv6 / IPv6
};

enum class TokenPolicy : int32_t {
    PlainText = 0,
    Signed = 1,
    Encrypted = 2,
};

enum class CqMode : int32_t {
    UserCtlNormal = 0,
    UserCtlInline = 1,
};

}  // namespace umc::comm
