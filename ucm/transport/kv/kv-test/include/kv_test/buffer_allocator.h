#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include "kv_test/kv_test_types.h"

namespace UC::KVTest {

std::size_t DeviceBufferAlignment();
std::size_t DeviceMrRegisterAlignment();
std::size_t AlignUp(std::size_t value, std::size_t alignment);
Status AllocateDeviceBuffer(std::size_t size, bool useAivRegistrationConstraints,
                            std::shared_ptr<void>& deviceBuffer);
Status CopyHostToDevice(const std::vector<std::uint8_t>& hostBuffer, std::uintptr_t deviceAddr,
                        const std::string& context);
UC::ASU::MemoryRegion MakeHostRegion(std::vector<std::uint8_t>& buffer);
UC::ASU::MemoryRegion MakeDeviceRegion(std::uint64_t addr, std::size_t size);
UC::ASU::KVBuffer MakeKvBuffer(const UC::ASU::CacheKey& key, const UC::ASU::MemoryRegion& region);

class BufferAllocator {
public:
    Status BuildStoreBuffers(const GeneratedData& data, PayloadBufferPlacement placement,
                             BufferSet& buffers) const;
    Status BuildRetrieveBuffers(const GeneratedData& data, PayloadBufferPlacement placement,
                                BufferSet& buffers) const;
    Status CopyDeviceBuffersToHost(BufferSet& buffers) const;
};

}  // namespace UC::KVTest
