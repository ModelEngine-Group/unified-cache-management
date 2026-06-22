#pragma once

#include "core/transport.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace transport {

struct ReceiveMessage {
    ManagerID manager_id;
    const void* data = nullptr;
    uint64_t length = 0;
};

using ReceiveHandler = std::function<Status(const ReceiveMessage& message)>;

namespace detail {

uint64_t PtrToU64(const void* ptr);
void* U64ToPtr(uint64_t value);

bool AppendU64(Metadata& out, uint64_t value);
bool ReadU64(const Metadata& input, size_t& offset, uint64_t& value);
bool AppendU32(Metadata& out, uint32_t value);
bool ReadU32(const Metadata& input, size_t& offset, uint32_t& value);
bool AppendU16(Metadata& out, uint16_t value);
bool ReadU16(const Metadata& input, size_t& offset, uint16_t& value);
bool AppendU8(Metadata& out, uint8_t value);
bool ReadU8(const Metadata& input, size_t& offset, uint8_t& value);
bool AppendString(Metadata& out, const std::string& value);
bool ReadString(const Metadata& input, size_t& offset, std::string& value);
bool AppendMetadataRecord(Metadata& out, const std::string& name, const Metadata& metadata);
bool ReadMetadataRecord(const Metadata& input, size_t& offset, std::string& name, Metadata& metadata);

}  // namespace detail
}  // namespace transport
