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
#include "control_codec.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace UC::Store {
namespace {

constexpr std::uint8_t kResultKindNone = 0;
constexpr std::uint8_t kResultKindHits = 1;
constexpr std::uint8_t kResultKindSlots = 2;

constexpr std::uint16_t kMaxShortStringLength = 65535;

bool IsDataOp(std::uint8_t op)
{
    return op >= static_cast<std::uint8_t>(ControlOp::Lookup) &&
           op <= static_cast<std::uint8_t>(ControlOp::Commit);
}

/** Fail-safe little-endian encoder; never fails -- all validity is checked up front. */
class Encoder {
public:
    void U8(std::uint8_t v) { buf_.push_back(static_cast<std::byte>(v)); }
    void U16(std::uint16_t v)
    {
        buf_.push_back(static_cast<std::byte>(v & 0xFF));
        buf_.push_back(static_cast<std::byte>((v >> 8) & 0xFF));
    }
    void U32(std::uint32_t v)
    {
        for (int shift = 0; shift <= 24; shift += 8) {
            buf_.push_back(static_cast<std::byte>((v >> shift) & 0xFF));
        }
    }
    void U64(std::uint64_t v)
    {
        for (int shift = 0; shift <= 56; shift += 8) {
            buf_.push_back(static_cast<std::byte>((v >> shift) & 0xFF));
        }
    }
    void I32(std::int32_t v) { U32(static_cast<std::uint32_t>(v)); }
    void I64(std::int64_t v) { U64(static_cast<std::uint64_t>(v)); }
    void Bytes(const std::byte* data, std::size_t len)
    {
        buf_.insert(buf_.end(), data, data + len);
    }
    void Block(const BlockId& id) { Bytes(id.bytes.data(), id.bytes.size()); }
    void Str(const std::string& s)
    {
        U16(static_cast<std::uint16_t>(s.size()));
        Bytes(reinterpret_cast<const std::byte*>(s.data()), s.size());
    }
    std::vector<std::byte> Take() { return std::move(buf_); }

private:
    std::vector<std::byte> buf_;
};

/**
 * @brief Fail-safe little-endian decoder: once a read runs past the end every
 *        subsequent read is a no-op and Failed() stays set, so length checks
 *        happen up front and malformed inputs never over-allocate.
 */
class Decoder {
public:
    explicit Decoder(const std::vector<std::byte>& data) : data_{data.data()}, size_{data.size()} {}

    std::uint8_t U8()
    {
        if (!Ensure(1)) { return 0; }
        return std::to_integer<std::uint8_t>(data_[pos_++]);
    }
    std::uint16_t U16()
    {
        if (!Ensure(2)) { return 0; }
        const std::uint16_t lo = std::to_integer<std::uint16_t>(data_[pos_]);
        const std::uint16_t hi = std::to_integer<std::uint16_t>(data_[pos_ + 1]);
        pos_ += 2;
        return static_cast<std::uint16_t>(lo | (hi << 8));
    }
    std::uint32_t U32()
    {
        std::uint32_t v = 0;
        for (int shift = 0; shift <= 24; shift += 8) {
            if (!Ensure(1)) { return 0; }
            v |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data_[pos_++])) << shift;
        }
        return v;
    }
    std::uint64_t U64()
    {
        std::uint64_t v = 0;
        for (int shift = 0; shift <= 56; shift += 8) {
            if (!Ensure(1)) { return 0; }
            v |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(data_[pos_++])) << shift;
        }
        return v;
    }
    std::int32_t I32() { return static_cast<std::int32_t>(U32()); }
    std::int64_t I64() { return static_cast<std::int64_t>(U64()); }

    std::vector<std::byte> Bytes(std::size_t len)
    {
        if (!Ensure(len)) { return {}; }
        std::vector<std::byte> out(data_ + pos_, data_ + pos_ + len);
        pos_ += len;
        return out;
    }

    BlockId Block()
    {
        BlockId id;
        if (!Ensure(id.bytes.size())) { return id; }
        std::memcpy(id.bytes.data(), data_ + pos_, id.bytes.size());
        pos_ += id.bytes.size();
        return id;
    }

    std::string Str()
    {
        const std::uint16_t len = U16();
        if (Failed()) { return {}; }
        std::vector<std::byte> raw = Bytes(len);
        if (Failed()) { return {}; }
        return std::string{reinterpret_cast<const char*>(raw.data()), raw.size()};
    }

    bool Failed() const { return failed_; }
    bool AtEnd() const { return pos_ == size_; }
    std::size_t Remaining() const { return size_ - pos_; }

private:
    bool Ensure(std::size_t need)
    {
        if (failed_ || size_ - pos_ < need) {
            failed_ = true;
            return false;
        }
        return true;
    }

    const std::byte* data_;
    std::size_t size_;
    std::size_t pos_{0};
    bool failed_{false};
};

Status TruncatedError()
{
    return Status::InvalidParam("control message is truncated or malformed");
}

}  // namespace

Expected<std::vector<std::byte>> EncodeRequest(const ControlRequest& request)
{
    const std::uint8_t op = static_cast<std::uint8_t>(request.op);
    if (!IsDataOp(op)) {
        return Status::InvalidParam("control request op {} is not a data-plane opcode", op);
    }

    Encoder enc;
    enc.U8(op);
    enc.U32(request.seq);
    enc.U32(static_cast<std::uint32_t>(request.blockIds.size()));
    for (const BlockId& id : request.blockIds) { enc.Block(id); }
    enc.U8(request.hasTtl ? 1 : 0);
    enc.I64(request.ttlMs);
    enc.U8(request.success ? 1 : 0);
    return enc.Take();
}

Expected<ControlRequest> DecodeRequest(const std::vector<std::byte>& data)
{
    Decoder dec{data};
    const std::uint8_t op = dec.U8();
    if (!dec.Failed() && !IsDataOp(op)) {
        return Status::InvalidParam("control request op {} is not a data-plane opcode", op);
    }

    ControlRequest request;
    request.op = static_cast<ControlOp>(op);
    request.seq = dec.U32();
    const std::uint32_t count = dec.U32();
    request.blockIds.reserve(std::min<std::uint64_t>(count, data.size() / sizeof(BlockId)));
    for (std::uint32_t i = 0; i < count && !dec.Failed(); ++i) {
        request.blockIds.push_back(dec.Block());
    }
    request.hasTtl = dec.U8() != 0;
    request.ttlMs = dec.I64();
    request.success = dec.U8() != 0;

    if (dec.Failed()) { return TruncatedError(); }
    if (!dec.AtEnd()) {
        return Status::InvalidParam("control request has {} trailing bytes", dec.Remaining());
    }
    return request;
}

Expected<std::vector<std::byte>> EncodeResponse(const ControlResponse& response)
{
    if (!response.hits.empty() && !response.slots.empty()) {
        return Status::InvalidParam("control response carries both hits and slots result bodies");
    }

    Encoder enc;
    enc.U32(response.seq);
    enc.U8(response.ok ? 1 : 0);
    enc.I32(response.statusCode);
    enc.U32(static_cast<std::uint32_t>(response.message.size()));
    enc.Bytes(reinterpret_cast<const std::byte*>(response.message.data()), response.message.size());
    if (!response.hits.empty()) {
        enc.U8(kResultKindHits);
        enc.U32(static_cast<std::uint32_t>(response.hits.size()));
        for (std::uint8_t hit : response.hits) { enc.U8(hit); }
    } else if (!response.slots.empty()) {
        enc.U8(kResultKindSlots);
        enc.U32(static_cast<std::uint32_t>(response.slots.size()));
        for (std::uint32_t slot : response.slots) { enc.U32(slot); }
    } else {
        enc.U8(kResultKindNone);
    }
    enc.U32(static_cast<std::uint32_t>(response.tokenPayload.size()));
    enc.Bytes(response.tokenPayload.data(), response.tokenPayload.size());
    return enc.Take();
}

Expected<ControlResponse> DecodeResponse(const std::vector<std::byte>& data)
{
    Decoder dec{data};
    ControlResponse response;
    response.seq = dec.U32();
    response.ok = dec.U8() != 0;
    response.statusCode = dec.I32();
    const std::uint32_t msgLen = dec.U32();
    std::vector<std::byte> message = dec.Bytes(msgLen);
    if (!dec.Failed()) {
        response.message.assign(reinterpret_cast<const char*>(message.data()), message.size());
    }

    const std::uint8_t kind = dec.U8();
    if (!dec.Failed() && kind != kResultKindNone && kind != kResultKindHits &&
        kind != kResultKindSlots) {
        return Status::InvalidParam("control response has unknown result kind {}", kind);
    }
    if (kind == kResultKindHits) {
        const std::uint32_t count = dec.U32();
        response.hits.reserve(
            std::min<std::uint32_t>(count, static_cast<std::uint32_t>(dec.Remaining())));
        for (std::uint32_t i = 0; i < count && !dec.Failed(); ++i) {
            response.hits.push_back(dec.U8());
        }
    } else if (kind == kResultKindSlots) {
        const std::uint32_t count = dec.U32();
        response.slots.reserve(
            std::min<std::uint32_t>(count, static_cast<std::uint32_t>(dec.Remaining() / 4)));
        for (std::uint32_t i = 0; i < count && !dec.Failed(); ++i) {
            response.slots.push_back(dec.U32());
        }
    }

    const std::uint32_t tokenLen = dec.U32();
    response.tokenPayload = dec.Bytes(tokenLen);

    if (dec.Failed()) { return TruncatedError(); }
    if (!dec.AtEnd()) {
        return Status::InvalidParam("control response has {} trailing bytes", dec.Remaining());
    }
    return response;
}

Expected<std::vector<std::byte>> EncodeHelloRequest(const HelloRequest& hello)
{
    if (hello.role.size() > kMaxShortStringLength ||
        hello.uniqueId.size() > kMaxShortStringLength ||
        hello.replacementPolicy.size() > kMaxShortStringLength) {
        return Status::InvalidParam("hello request string field exceeds {} bytes",
                                    kMaxShortStringLength);
    }

    Encoder enc;
    enc.U8(static_cast<std::uint8_t>(ControlOp::Hello));
    enc.U32(0);
    enc.Str(hello.role);
    enc.Str(hello.uniqueId);
    enc.U64(hello.blockTotalNumber);
    enc.U64(hello.blockSizeBytes);
    enc.Str(hello.replacementPolicy);
    enc.U8(hello.forceHugePage ? 1 : 0);
    return enc.Take();
}

Expected<HelloRequest> DecodeHelloRequest(const std::vector<std::byte>& data)
{
    Decoder dec{data};
    const std::uint8_t op = dec.U8();
    if (!dec.Failed() && op != static_cast<std::uint8_t>(ControlOp::Hello)) {
        return Status::InvalidParam("hello request expected op Hello, got {}", op);
    }
    (void)dec.U32();  // sequence number: reserved by the handshake, kept on the wire
    HelloRequest hello;
    hello.role = dec.Str();
    hello.uniqueId = dec.Str();
    hello.blockTotalNumber = dec.U64();
    hello.blockSizeBytes = dec.U64();
    hello.replacementPolicy = dec.Str();
    hello.forceHugePage = dec.U8() != 0;

    if (dec.Failed()) { return TruncatedError(); }
    if (!dec.AtEnd()) {
        return Status::InvalidParam("hello request has {} trailing bytes", dec.Remaining());
    }
    return hello;
}

}  // namespace UC::Store
