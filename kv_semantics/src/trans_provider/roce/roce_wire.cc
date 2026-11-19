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
#include "roce_wire.h"
#include <array>
#include <cstring>

namespace kv {
namespace roce {
namespace {

// Offsets inside the 160 byte body.
constexpr std::size_t kCapOffset = 0;
constexpr std::size_t kRsvOffset = 4;
constexpr std::size_t kRsvBytes = 24;
constexpr std::size_t kPrivateLenOffset = 28;
constexpr std::size_t kPrivateOffset = 32;
constexpr std::size_t kRequestPrivateBytes = 4;
// The reply grew by delete_key_batch between minor versions: 36 bytes is the
// older layout, 38 the current one, and a larger value means the peer sends
// fields this build does not know about yet.
constexpr std::size_t kReplyMinPrivateBytes = 36;
constexpr std::size_t kReplyPrivateBytes = 38;

// Offsets inside the connect request body.
constexpr std::size_t kGidOffset = 0;
constexpr std::size_t kLidOffset = 16;
constexpr std::size_t kMtuOffset = 18;
constexpr std::size_t kTotalQpOffset = 19;
constexpr std::size_t kSlOffset = 20;
constexpr std::size_t kTrafficClassOffset = 21;
constexpr std::size_t kRnrTimerOffset = 22;
constexpr std::size_t kRnrRetryOffset = 23;
constexpr std::size_t kTimeoutOffset = 24;
constexpr std::size_t kRetryCntOffset = 25;
constexpr std::size_t kQpRdAtomOffset = 26;
constexpr std::size_t kStartPsnOffset = 28;
constexpr std::size_t kQpnOffset = 32;

const std::array<std::uint32_t, 256>& CrcTable()
{
    // Reflected CRC-32C (Castagnoli, polynomial 0x1EDC6F41) lookup table.
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> values{};
        for (std::uint32_t index = 0; index < 256; ++index) {
            std::uint32_t value = index;
            for (int bit = 0; bit < 8; ++bit) {
                value = (value & 1U) != 0U ? (0x82F63B78U ^ (value >> 1U)) : (value >> 1U);
            }
            values[index] = value;
        }
        return values;
    }();
    return table;
}

class Writer {
public:
    explicit Writer(std::vector<std::uint8_t>& buffer) : buffer_(buffer) {}

    void U8(std::uint8_t value) { buffer_.push_back(value); }

    void U16(std::uint16_t value)
    {
        buffer_.push_back(static_cast<std::uint8_t>(value >> 8U));
        buffer_.push_back(static_cast<std::uint8_t>(value));
    }

    void U32(std::uint32_t value)
    {
        buffer_.push_back(static_cast<std::uint8_t>(value >> 24U));
        buffer_.push_back(static_cast<std::uint8_t>(value >> 16U));
        buffer_.push_back(static_cast<std::uint8_t>(value >> 8U));
        buffer_.push_back(static_cast<std::uint8_t>(value));
    }

    // The 128 byte private area of the capability messages is the one part
    // of the protocol whose values travel little endian.
    void U16Le(std::uint16_t value)
    {
        buffer_.push_back(static_cast<std::uint8_t>(value));
        buffer_.push_back(static_cast<std::uint8_t>(value >> 8U));
    }

    void U32Le(std::uint32_t value)
    {
        buffer_.push_back(static_cast<std::uint8_t>(value));
        buffer_.push_back(static_cast<std::uint8_t>(value >> 8U));
        buffer_.push_back(static_cast<std::uint8_t>(value >> 16U));
        buffer_.push_back(static_cast<std::uint8_t>(value >> 24U));
    }

    void Bytes(const std::uint8_t* data, std::size_t size)
    {
        buffer_.insert(buffer_.end(), data, data + size);
    }

    void Zeroes(std::size_t size) { buffer_.insert(buffer_.end(), size, 0U); }

    void PatchU32(std::size_t offset, std::uint32_t value)
    {
        buffer_[offset] = static_cast<std::uint8_t>(value >> 24U);
        buffer_[offset + 1] = static_cast<std::uint8_t>(value >> 16U);
        buffer_[offset + 2] = static_cast<std::uint8_t>(value >> 8U);
        buffer_[offset + 3] = static_cast<std::uint8_t>(value);
    }

private:
    std::vector<std::uint8_t>& buffer_;
};

class Reader {
public:
    Reader(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

    bool U8(std::uint8_t& value)
    {
        if (offset_ + 1 > size_) { return false; }
        value = data_[offset_++];
        return true;
    }

    bool U16(std::uint16_t& value)
    {
        if (offset_ + 2 > size_) { return false; }
        value = static_cast<std::uint16_t>((static_cast<std::uint16_t>(data_[offset_]) << 8U) |
                                           static_cast<std::uint16_t>(data_[offset_ + 1]));
        offset_ += 2;
        return true;
    }

    bool U16Le(std::uint16_t& value)
    {
        if (offset_ + 2 > size_) { return false; }
        value = static_cast<std::uint16_t>(static_cast<std::uint16_t>(data_[offset_]) |
                                           (static_cast<std::uint16_t>(data_[offset_ + 1]) << 8U));
        offset_ += 2;
        return true;
    }

    bool U32(std::uint32_t& value)
    {
        if (offset_ + 4 > size_) { return false; }
        value = (static_cast<std::uint32_t>(data_[offset_]) << 24U) |
                (static_cast<std::uint32_t>(data_[offset_ + 1]) << 16U) |
                (static_cast<std::uint32_t>(data_[offset_ + 2]) << 8U) |
                static_cast<std::uint32_t>(data_[offset_ + 3]);
        offset_ += 4;
        return true;
    }

    bool U32Le(std::uint32_t& value)
    {
        if (offset_ + 4 > size_) { return false; }
        value = static_cast<std::uint32_t>(data_[offset_]) |
                (static_cast<std::uint32_t>(data_[offset_ + 1]) << 8U) |
                (static_cast<std::uint32_t>(data_[offset_ + 2]) << 16U) |
                (static_cast<std::uint32_t>(data_[offset_ + 3]) << 24U);
        offset_ += 4;
        return true;
    }

    bool Skip(std::size_t size)
    {
        if (offset_ + size > size_) { return false; }
        offset_ += size;
        return true;
    }

    bool Seek(std::size_t offset)
    {
        if (offset > size_) { return false; }
        offset_ = offset;
        return true;
    }

    bool Bytes(std::uint8_t* destination, std::size_t size)
    {
        if (offset_ + size > size_) { return false; }
        std::memcpy(destination, data_ + offset_, size);
        offset_ += size;
        return true;
    }

    std::size_t Offset() const { return offset_; }

private:
    const std::uint8_t* data_;
    std::size_t size_;
    std::size_t offset_{0};
};

Status Fail(StatusCode code, const std::string& message)
{
    return Status::Error(code, "roce wire: " + message);
}

void FinishFrame(std::vector<std::uint8_t>& frame, std::uint8_t cmd)
{
    // Header: crc(4) ver(1) cmd(1) pad(6) len(4).
    frame[0] = 0;
    frame[1] = 0;
    frame[2] = 0;
    frame[3] = 0;
    frame[4] = kWireVersion;
    frame[5] = cmd;
    const std::uint32_t bodyLength = static_cast<std::uint32_t>(frame.size() - kWireHeaderBytes);
    frame[12] = static_cast<std::uint8_t>(bodyLength >> 24U);
    frame[13] = static_cast<std::uint8_t>(bodyLength >> 16U);
    frame[14] = static_cast<std::uint8_t>(bodyLength >> 8U);
    frame[15] = static_cast<std::uint8_t>(bodyLength);
    // The checksum covers everything except the crc field itself, so the length
    // has to be in place before it is computed.
    const std::uint32_t crc = Crc32c(frame.data() + 4, frame.size() - 4);
    frame[0] = static_cast<std::uint8_t>(crc >> 24U);
    frame[1] = static_cast<std::uint8_t>(crc >> 16U);
    frame[2] = static_cast<std::uint8_t>(crc >> 8U);
    frame[3] = static_cast<std::uint8_t>(crc);
}

}  // namespace

std::uint32_t Crc32c(const std::uint8_t* data, std::size_t size)
{
    // Reflected CRC-32C with a zero initial value and no final inversion,
    // so the checksum of an empty buffer is zero.
    std::uint32_t crc = 0;
    const auto& table = CrcTable();
    for (std::size_t index = 0; index < size; ++index) {
        crc = table[(crc ^ data[index]) & 0xFFU] ^ (crc >> 8U);
    }
    return crc;
}

std::uint8_t MtuCodeFromBytes(std::size_t mtu)
{
    switch (mtu) {
        case 256: return 1;
        case 512: return 2;
        case 1024: return 3;
        case 2048: return 4;
        case 4096: return 5;
        default: return 5;
    }
}

std::size_t MtuBytesFromCode(std::uint8_t code)
{
    switch (code) {
        case 1: return 256;
        case 2: return 512;
        case 3: return 1024;
        case 4: return 2048;
        case 5: return 4096;
        default: return 0;
    }
}

Status EncodeCapabilityRequest(const CapabilityRequest& request, std::vector<std::uint8_t>& frame)
{
    frame.assign(kWireHeaderBytes, 0U);
    Writer writer(frame);
    writer.U32(0);                     // cap
    writer.Zeroes(kRsvBytes);          // rsv
    writer.U32(kRequestPrivateBytes);  // private_len
    writer.U8(request.majorVersion);
    writer.U8(request.minorVersion);
    writer.U16Le(request.kato);
    writer.Zeroes(kWireBodyBytes - (kPrivateOffset + kRequestPrivateBytes));
    FinishFrame(frame, kWireCmdCapability);
    return Status::OK();
}

Status EncodeCapabilityReply(const Capabilities& capabilities, std::vector<std::uint8_t>& frame)
{
    frame.assign(kWireHeaderBytes, 0U);
    Writer writer(frame);
    writer.U32(0);                   // cap
    writer.Zeroes(kRsvBytes);        // rsv
    writer.U32(kReplyPrivateBytes);  // private_len
    writer.U8(capabilities.majorVersion);
    writer.U8(capabilities.minorVersion);
    writer.U16Le(capabilities.queueNum);
    writer.U16Le(capabilities.adminqDepth);
    writer.U16Le(capabilities.ioqDepth);
    writer.U16Le(capabilities.ioqKeyBatch);
    writer.U16Le(capabilities.totalKeyBatch);
    writer.U32Le(capabilities.srMdts);
    writer.U32Le(capabilities.bsrMdts);
    writer.U16Le(capabilities.storeKeyBatch);
    writer.U16Le(capabilities.retrieveKeyBatch);
    writer.U16Le(capabilities.deleteKeyBatch);
    writer.U16Le(capabilities.existKeyBatch);
    writer.U8(capabilities.keyLength);
    writer.U8(0);  // pad
    writer.U16Le(capabilities.controllerId);
    writer.U32Le(capabilities.controllerCap);
    writer.U16Le(0);  // pad, private payload is 38 bytes
    writer.Zeroes(kWireBodyBytes - (kPrivateOffset + kReplyPrivateBytes));
    FinishFrame(frame, kWireCmdCapability);
    return Status::OK();
}

Status EncodeConnectRequest(std::uint8_t cmd, const QpInfo& info, std::vector<std::uint8_t>& frame)
{
    if (cmd != kWireCmdConnectRequest && cmd != kWireCmdConnectResponse) {
        return Fail(StatusCode::INVALID_ARGUMENT, "connect request needs cmd 1 or 2");
    }
    if (info.totalQpNum == 0) {
        return Fail(StatusCode::INVALID_ARGUMENT, "connect request needs at least one queue pair");
    }
    if (info.totalQpNum > kMaxQpPerLink) {
        return Fail(StatusCode::INVALID_ARGUMENT, "too many queue pairs");
    }
    frame.assign(kWireHeaderBytes, 0U);
    Writer writer(frame);
    writer.Bytes(info.gid, sizeof(info.gid));
    writer.U16(info.lid);
    writer.U8(info.mtu);
    writer.U8(info.totalQpNum);
    writer.U8(info.sl);
    writer.U8(info.trafficClass);
    writer.U8(info.rnrTimer);
    writer.U8(info.rnrRetryCnt);
    writer.U8(info.timeout);
    writer.U8(info.retryCnt);
    writer.U8(info.qpRdAtom);
    writer.U8(0);  // pad
    writer.U32(info.startPsn);
    // Only the queue pairs this side created travel on the wire: a fixed 32
    // entry table would make every reader consume numbers that were never set.
    for (std::size_t index = 0; index < info.totalQpNum; ++index) { writer.U32(info.qpn[index]); }
    FinishFrame(frame, cmd);
    return Status::OK();
}

Status EncodeConnectDone(std::vector<std::uint8_t>& frame)
{
    frame.assign(kWireHeaderBytes, 0U);
    FinishFrame(frame, kWireCmdConnectDone);
    return Status::OK();
}

Status EncodeDisconnect(std::vector<std::uint8_t>& frame)
{
    frame.assign(kWireHeaderBytes, 0U);
    FinishFrame(frame, kWireCmdDisconnect);
    return Status::OK();
}

bool PeekBodyLength(const std::uint8_t* frame, std::size_t size, std::uint32_t& bodyLength)
{
    if (frame == nullptr || size < kWireHeaderBytes) { return false; }
    Reader reader(frame + 12, size - 12);
    return reader.U32(bodyLength);
}

Status DecodeMessage(const std::uint8_t* frame, std::size_t size, Message& message,
                     std::string& error)
{
    error.clear();
    if (size < kWireHeaderBytes) {
        error = "frame shorter than the header";
        return Fail(StatusCode::INVALID_ARGUMENT, error);
    }

    std::uint32_t crc = 0;
    if (!Reader(frame, size).U32(crc)) {
        error = "header is truncated";
        return Fail(StatusCode::INVALID_ARGUMENT, error);
    }
    const std::uint8_t version = frame[4];
    const std::uint8_t cmd = frame[kWireCmdOffset];
    std::uint32_t bodyLength = 0;
    if (!PeekBodyLength(frame, size, bodyLength)) {
        error = "header is truncated";
        return Fail(StatusCode::INVALID_ARGUMENT, error);
    }

    if (version != kWireVersion) {
        error = "unsupported message version " + std::to_string(version);
        return Fail(StatusCode::INVALID_ARGUMENT, error);
    }
    if (bodyLength > kWireBodyBytes || bodyLength != size - kWireHeaderBytes) {
        error = "body length " + std::to_string(bodyLength) + " does not match the frame";
        return Fail(StatusCode::INVALID_ARGUMENT, error);
    }
    const std::uint32_t expected = Crc32c(frame + 4, size - 4);
    if (crc != expected) {
        error = "checksum mismatch";
        return Fail(StatusCode::INVALID_ARGUMENT, error);
    }

    const std::uint8_t* body = frame + kWireHeaderBytes;
    message = Message{};
    message.cmd = cmd;

    switch (cmd) {
        case kWireCmdCapability: {
            std::uint32_t privateLength = 0;
            Reader reader(body, bodyLength);
            if (!reader.Skip(kPrivateLenOffset) || !reader.U32(privateLength)) {
                error = "capability message is truncated";
                return Fail(StatusCode::INVALID_ARGUMENT, error);
            }
            if (privateLength == kRequestPrivateBytes) {
                // A responder has to reject a peer it cannot speak to before it
                // answers, so the request version is kept.
                Reader privateReader(body + kPrivateOffset, bodyLength - kPrivateOffset);
                if (!privateReader.U8(message.requestMajorVersion) ||
                    !privateReader.U8(message.requestMinorVersion)) {
                    error = "capability request is truncated";
                    return Fail(StatusCode::INVALID_ARGUMENT, error);
                }
                return Status::OK();
            }
            if (privateLength >= kReplyMinPrivateBytes) {
                Reader privateReader(body + kPrivateOffset, bodyLength - kPrivateOffset);
                Capabilities capabilities;
                std::uint8_t pad = 0;
                // Anything past the fields the peer advertises is ignored, so a
                // newer peer can append fields without breaking this build.
                if (!privateReader.U8(capabilities.majorVersion) ||
                    !privateReader.U8(capabilities.minorVersion) ||
                    !privateReader.U16Le(capabilities.queueNum) ||
                    !privateReader.U16Le(capabilities.adminqDepth) ||
                    !privateReader.U16Le(capabilities.ioqDepth) ||
                    !privateReader.U16Le(capabilities.ioqKeyBatch) ||
                    !privateReader.U16Le(capabilities.totalKeyBatch) ||
                    !privateReader.U32Le(capabilities.srMdts) ||
                    !privateReader.U32Le(capabilities.bsrMdts) ||
                    !privateReader.U16Le(capabilities.storeKeyBatch) ||
                    !privateReader.U16Le(capabilities.retrieveKeyBatch)) {
                    error = "capability reply is truncated";
                    return Fail(StatusCode::INVALID_ARGUMENT, error);
                }
                // delete_key_batch only exists from the 38 byte layout on.
                if (privateLength >= kReplyPrivateBytes &&
                    !privateReader.U16Le(capabilities.deleteKeyBatch)) {
                    error = "capability reply is truncated";
                    return Fail(StatusCode::INVALID_ARGUMENT, error);
                }
                if (!privateReader.U16Le(capabilities.existKeyBatch) ||
                    !privateReader.U8(capabilities.keyLength) || !privateReader.U8(pad) ||
                    !privateReader.U16Le(capabilities.controllerId) ||
                    !privateReader.U32Le(capabilities.controllerCap)) {
                    error = "capability reply is truncated";
                    return Fail(StatusCode::INVALID_ARGUMENT, error);
                }
                message.capabilities = capabilities;
                message.capabilityReply = true;
                return Status::OK();
            }
            error = "unexpected capability private length " + std::to_string(privateLength);
            return Fail(StatusCode::INVALID_ARGUMENT, error);
        }
        case kWireCmdConnectRequest:
        case kWireCmdConnectResponse: {
            Reader reader(body, bodyLength);
            QpInfo info;
            std::uint8_t pad = 0;
            if (!reader.Bytes(info.gid, sizeof(info.gid)) || !reader.U16(info.lid) ||
                !reader.U8(info.mtu) || !reader.U8(info.totalQpNum) || !reader.U8(info.sl) ||
                !reader.U8(info.trafficClass) || !reader.U8(info.rnrTimer) ||
                !reader.U8(info.rnrRetryCnt) || !reader.U8(info.timeout) ||
                !reader.U8(info.retryCnt) || !reader.U8(info.qpRdAtom) || !reader.U8(pad) ||
                !reader.U32(info.startPsn)) {
                error = "connect request is truncated";
                return Fail(StatusCode::INVALID_ARGUMENT, error);
            }
            if (info.totalQpNum > kMaxQpPerLink) {
                error = "connect request carries too many queue pairs";
                return Fail(StatusCode::INVALID_ARGUMENT, error);
            }
            for (std::size_t index = 0; index < info.totalQpNum; ++index) {
                if (!reader.U32(info.qpn[index])) {
                    error = "connect request queue pair table is truncated";
                    return Fail(StatusCode::INVALID_ARGUMENT, error);
                }
            }
            if (reader.Offset() != bodyLength) {
                error = "connect request carries " + std::to_string(bodyLength) +
                        " body bytes for " + std::to_string(info.totalQpNum) + " queue pairs";
                return Fail(StatusCode::INVALID_ARGUMENT, error);
            }
            message.qp = info;
            return Status::OK();
        }
        case kWireCmdConnectDone:
        case kWireCmdDisconnect: {
            if (bodyLength != 0) {
                error = "command " + std::to_string(cmd) + " must not carry a body";
                return Fail(StatusCode::INVALID_ARGUMENT, error);
            }
            return Status::OK();
        }
        default:
            error = "unknown command " + std::to_string(cmd);
            return Fail(StatusCode::INVALID_ARGUMENT, error);
    }
}

}  // namespace roce
}  // namespace kv
