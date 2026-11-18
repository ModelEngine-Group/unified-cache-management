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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "kv_types.h"

namespace kv {
namespace roce {

// Wire protocol exchanged over the out-of-band TCP channel.  The layout
// follows the link specification:
//
//   table 0  message header, 16 bytes
//   table 1  capability request  (cmd = 0), body 160 bytes
//   table 2  capability reply    (cmd = 0), body 160 bytes
//   table 3  connect request     (cmd = 1 initiator / 2 responder),
//            body 32 + 4 * total_qp_num (one 32 bit qpn per queue pair)
//   table 4  connect done        (cmd = 3), body 0
//   table 5  disconnect          (cmd = 4), body 0
//
// Multi-byte fields travel in big endian order, with one exception: the 128
// byte private area of the capability messages carries little endian values.
// The header checksum is CRC-32C (Castagnoli) over the header with the crc
// field itself skipped plus the whole body, with a zero initial value and no
// final inversion.

constexpr std::uint8_t kWireVersion = 1;

constexpr std::size_t kWireHeaderBytes = 16;

// Byte offset of the command inside the header: crc(4), version(1), command(1).
constexpr std::size_t kWireCmdOffset = 5;
// Longest body of any message; a connect request grows with its queue pairs
// and stops at 32 + 4 * kMaxQpPerLink.
constexpr std::size_t kWireBodyBytes = 160;
constexpr std::size_t kWireMaxFrameBytes = kWireHeaderBytes + kWireBodyBytes;

constexpr std::uint8_t kWireCmdCapability = 0;
constexpr std::uint8_t kWireCmdConnectRequest = 1;
constexpr std::uint8_t kWireCmdConnectResponse = 2;
constexpr std::uint8_t kWireCmdConnectDone = 3;
constexpr std::uint8_t kWireCmdDisconnect = 4;

// table 3 carries qpn[32].
constexpr std::size_t kMaxQpPerLink = 32;

// table 1 private payload: major/minor/kato.
struct CapabilityRequest {
    std::uint8_t majorVersion{1};
    std::uint8_t minorVersion{0};
    std::uint16_t kato{0};
};

// table 2 private payload.
struct Capabilities {
    std::uint8_t majorVersion{1};
    std::uint8_t minorVersion{0};
    std::uint16_t queueNum{1};
    std::uint16_t adminqDepth{0};
    std::uint16_t ioqDepth{256};
    std::uint16_t ioqKeyBatch{1};
    std::uint16_t totalKeyBatch{1};
    std::uint32_t srMdts{1U << 20U};
    std::uint32_t bsrMdts{1U << 20U};
    std::uint16_t storeKeyBatch{1};
    std::uint16_t retrieveKeyBatch{1};
    std::uint16_t deleteKeyBatch{1};
    std::uint16_t existKeyBatch{1};
    std::uint8_t keyLength{8};
    std::uint16_t controllerId{0};
    std::uint32_t controllerCap{0};
};

// table 3 payload.
struct QpInfo {
    std::uint8_t gid[16]{};
    std::uint16_t lid{0};
    std::uint8_t mtu{0};
    std::uint8_t totalQpNum{0};
    std::uint8_t sl{3};
    std::uint8_t trafficClass{104};
    std::uint8_t rnrTimer{12};
    std::uint8_t rnrRetryCnt{7};
    std::uint8_t timeout{17};
    std::uint8_t retryCnt{7};
    std::uint8_t qpRdAtom{16};
    std::uint32_t startPsn{0};
    std::uint32_t qpn[kMaxQpPerLink]{};
};

// Decoded frame; only the member that matches cmd carries data.
struct Message {
    std::uint8_t cmd{0};
    CapabilityRequest request{};
    Capabilities capabilities{};
    QpInfo qp{};
    // True when a cmd 0 message carries the reply form (table 2).
    bool capabilityReply{false};
};

// The MTU code used by table 3 (1 -> 256 ... 5 -> 4096).
std::uint8_t MtuCodeFromBytes(std::size_t mtu);

std::size_t MtuBytesFromCode(std::uint8_t code);

Status EncodeCapabilityRequest(const CapabilityRequest& request, std::vector<std::uint8_t>& frame);

Status EncodeCapabilityReply(const Capabilities& capabilities, std::vector<std::uint8_t>& frame);

Status EncodeConnectRequest(std::uint8_t cmd, const QpInfo& info, std::vector<std::uint8_t>& frame);

Status EncodeConnectDone(std::vector<std::uint8_t>& frame);

Status EncodeDisconnect(std::vector<std::uint8_t>& frame);

// Reads the announced body length out of a 16 byte header.
bool PeekBodyLength(const std::uint8_t* frame, std::size_t size, std::uint32_t& bodyLength);

Status DecodeMessage(const std::uint8_t* frame, std::size_t size, Message& message,
                     std::string& error);

}  // namespace roce
}  // namespace kv
