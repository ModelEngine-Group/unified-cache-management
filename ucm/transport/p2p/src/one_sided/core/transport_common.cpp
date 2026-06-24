#include "transport_common.h"
#include <climits>

namespace transport::detail {

uint64_t PtrToU64(const void* ptr)
{
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}

void* U64ToPtr(uint64_t value) { return reinterpret_cast<void*>(static_cast<uintptr_t>(value)); }

bool AppendU64(Metadata& out, uint64_t value)
{
    for (int shift = 56; shift >= 0; shift -= 8) {
        out.push_back(static_cast<uint8_t>((value >> shift) & 0xff));
    }
    return true;
}

bool ReadU64(const Metadata& input, size_t& offset, uint64_t& value)
{
    if (offset > input.size() || input.size() - offset < sizeof(uint64_t)) { return false; }
    value = 0;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) { value = (value << 8) | input[offset + i]; }
    offset += sizeof(uint64_t);
    return true;
}

bool AppendU32(Metadata& out, uint32_t value)
{
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>(value & 0xff));
    return true;
}

bool ReadU32(const Metadata& input, size_t& offset, uint32_t& value)
{
    if (offset > input.size() || input.size() - offset < sizeof(uint32_t)) { return false; }
    value = (static_cast<uint32_t>(input[offset]) << 24) |
            (static_cast<uint32_t>(input[offset + 1]) << 16) |
            (static_cast<uint32_t>(input[offset + 2]) << 8) |
            static_cast<uint32_t>(input[offset + 3]);
    offset += sizeof(uint32_t);
    return true;
}

bool AppendU16(Metadata& out, uint16_t value)
{
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>(value & 0xff));
    return true;
}

bool ReadU16(const Metadata& input, size_t& offset, uint16_t& value)
{
    if (offset > input.size() || input.size() - offset < sizeof(uint16_t)) { return false; }
    value = static_cast<uint16_t>((static_cast<uint16_t>(input[offset]) << 8) |
                                  static_cast<uint16_t>(input[offset + 1]));
    offset += sizeof(uint16_t);
    return true;
}

bool AppendU8(Metadata& out, uint8_t value)
{
    out.push_back(value);
    return true;
}

bool ReadU8(const Metadata& input, size_t& offset, uint8_t& value)
{
    if (offset >= input.size()) { return false; }
    value = input[offset++];
    return true;
}

bool AppendMetadata(Metadata& out, const Metadata& value)
{
    if (value.size() > UINT32_MAX) { return false; }
    AppendU32(out, static_cast<uint32_t>(value.size()));
    out.insert(out.end(), value.begin(), value.end());
    return true;
}

bool ReadMetadata(const Metadata& input, size_t& offset, Metadata& value)
{
    uint32_t size = 0;
    if (!ReadU32(input, offset, size) || offset > input.size() || input.size() - offset < size) {
        return false;
    }
    value.assign(input.begin() + static_cast<std::ptrdiff_t>(offset),
                 input.begin() + static_cast<std::ptrdiff_t>(offset + size));
    offset += size;
    return true;
}

bool AppendString(Metadata& out, const std::string& value)
{
    if (value.size() > UINT32_MAX) { return false; }
    AppendU32(out, static_cast<uint32_t>(value.size()));
    out.insert(out.end(), value.begin(), value.end());
    return true;
}

bool ReadString(const Metadata& input, size_t& offset, std::string& value)
{
    uint32_t size = 0;
    if (!ReadU32(input, offset, size) || offset > input.size() || input.size() - offset < size) {
        return false;
    }
    value.assign(input.begin() + static_cast<std::ptrdiff_t>(offset),
                 input.begin() + static_cast<std::ptrdiff_t>(offset + size));
    offset += size;
    return true;
}

bool AppendMetadataRecord(Metadata& out, const std::string& name, const Metadata& metadata)
{
    return AppendString(out, name) && AppendMetadata(out, metadata);
}

bool ReadMetadataRecord(const Metadata& input, size_t& offset, std::string& name,
                        Metadata& metadata)
{
    return ReadString(input, offset, name) && ReadMetadata(input, offset, metadata);
}

}  // namespace transport::detail
