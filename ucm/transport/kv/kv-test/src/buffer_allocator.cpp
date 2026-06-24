#include "kv_test/buffer_allocator.h"
#include <acl/acl.h>
#include <limits>

namespace UC::KVTest {

namespace {

constexpr int kExitInvalidArgument = 1;

constexpr std::uint8_t kRetrieveBufferInitialValue = 0xA5;
constexpr std::size_t kDeviceBufferAlignment = UC::ASU::kAsuAlignmentBytes;
constexpr std::size_t kDeviceMrRegisterAlignment = 2ULL * 1024ULL * 1024ULL;

UC::ASU::MemoryRegion MakeHostRegion(std::vector<std::uint8_t>& buffer)
{
    UC::ASU::MemoryRegion region;
    region.memoryType = UC::ASU::MemoryType::HOST;
    region.addr = buffer.empty() ? 0 : reinterpret_cast<std::uint64_t>(buffer.data());
    region.size = buffer.size();
    region.deviceId = -1;
    region.numaNode = -1;
    return region;
}

std::size_t AlignUp(std::size_t value, std::size_t alignment)
{
    if (alignment == 0) { return value; }
    const auto remainder = value % alignment;
    if (remainder == 0) { return value; }
    return value + alignment - remainder;
}

UC::ASU::MemoryRegion MakeDeviceRegion(std::uint64_t addr, std::size_t size)
{
    UC::ASU::MemoryRegion region;
    region.memoryType = UC::ASU::MemoryType::ASCEND_DEVICE;
    region.addr = addr;
    region.size = size;
    region.deviceId = 0;
    region.numaNode = -1;
    return region;
}

UC::ASU::KVBuffer MakeKvBuffer(const UC::ASU::CacheKey& key, const UC::ASU::MemoryRegion& region)
{
    UC::ASU::Buffer buffer;
    buffer.region = region;
    buffer.handle = UC::ASU::kInvalidMRHandle;
    return UC::ASU::KVBuffer{key, buffer};
}

Status ValidateGeneratedData(const GeneratedData& data, const std::string& operation)
{
    if (data.keys.size() != data.values.size()) {
        return Status::Error(kExitInvalidArgument,
                             operation + " generated key/value count mismatch");
    }
    return Status::Success();
}

Status CopyHostToDevice(const std::vector<std::uint8_t>& hostBuffer, void* deviceAddr)
{
    if (hostBuffer.empty()) { return Status::Success(); }

    const auto ret = aclrtMemcpy(deviceAddr, hostBuffer.size(), hostBuffer.data(),
                                 hostBuffer.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return Status::Error(kExitInvalidArgument,
                             "device payload host-to-device copy failed: size=" +
                                 std::to_string(hostBuffer.size()) + " ret=" + std::to_string(ret));
    }
    return Status::Success();
}

Status BuildDeviceBuffers(BufferSet& buffers)
{
    std::size_t totalSize = 0;
    buffers.deviceBufferOffsets.clear();
    buffers.deviceBufferOffsets.reserve(buffers.ownedBuffers.size());
    for (const auto& hostBuffer : buffers.ownedBuffers) {
        const auto offset = AlignUp(totalSize, kDeviceBufferAlignment);
        if (offset < totalSize ||
            hostBuffer.size() > std::numeric_limits<std::size_t>::max() - offset) {
            return Status::Error(kExitInvalidArgument, "device payload buffer size overflow");
        }
        buffers.deviceBufferOffsets.emplace_back(offset);
        totalSize = offset + hostBuffer.size();
    }

    if (totalSize == 0) { return Status::Success(); }

    const auto registerSize = AlignUp(totalSize, kDeviceMrRegisterAlignment);
    if (registerSize < totalSize) {
        return Status::Error(kExitInvalidArgument, "device payload register size overflow");
    }

    void* ptr = nullptr;
    auto ret = aclrtMalloc(&ptr, registerSize, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    if (ret != ACL_SUCCESS) {
        return Status::Error(kExitInvalidArgument, "device payload aclrtMalloc failed: size=" +
                                                       std::to_string(registerSize) +
                                                       " ret=" + std::to_string(ret));
    }

    auto deviceBuffer = std::shared_ptr<void>(ptr, aclrtFree);
    const auto baseAddr = reinterpret_cast<std::uintptr_t>(deviceBuffer.get());
    for (std::size_t index = 0; index < buffers.ownedBuffers.size(); ++index) {
        auto* deviceAddr = reinterpret_cast<void*>(baseAddr + buffers.deviceBufferOffsets[index]);
        auto status = CopyHostToDevice(buffers.ownedBuffers[index], deviceAddr);
        if (!status.Ok()) { return status; }
    }

    buffers.deviceBuffers.emplace_back(deviceBuffer);
    buffers.regions.emplace_back(MakeDeviceRegion(baseAddr, registerSize));
    buffers.entryRegionIndexes.assign(buffers.ownedBuffers.size(), 0);
    return Status::Success();
}

UC::ASU::MemoryRegion MakeRegion(BufferSet& buffers, std::size_t index,
                                 PayloadBufferPlacement placement)
{
    if (placement == PayloadBufferPlacement::ASCEND_DEVICE) {
        const auto baseAddr =
            buffers.deviceBuffers.empty()
                ? 0
                : reinterpret_cast<std::uintptr_t>(buffers.deviceBuffers.front().get());
        return MakeDeviceRegion(baseAddr + buffers.deviceBufferOffsets[index],
                                buffers.ownedBuffers[index].size());
    }
    return MakeHostRegion(buffers.ownedBuffers[index]);
}

}  // namespace

Status BufferAllocator::BuildStoreBuffers(const GeneratedData& data,
                                          PayloadBufferPlacement placement,
                                          BufferSet& buffers) const
{
    auto status = ValidateGeneratedData(data, "store");
    if (!status.Ok()) { return status; }

    buffers = BufferSet{};
    buffers.ownedBuffers.reserve(data.values.size());
    buffers.regions.reserve(data.values.size());
    buffers.entries.reserve(data.values.size());

    for (const auto& value : data.values) { buffers.ownedBuffers.emplace_back(value); }
    if (placement == PayloadBufferPlacement::ASCEND_DEVICE) {
        status = BuildDeviceBuffers(buffers);
        if (!status.Ok()) { return status; }
    }

    for (std::size_t index = 0; index < data.keys.size(); ++index) {
        auto region = MakeRegion(buffers, index, placement);
        if (placement != PayloadBufferPlacement::ASCEND_DEVICE) {
            buffers.regions.emplace_back(region);
        }
        buffers.entries.emplace_back(MakeKvBuffer(data.keys[index], region));
    }

    return Status::Success();
}

Status BufferAllocator::BuildRetrieveBuffers(const GeneratedData& data,
                                             PayloadBufferPlacement placement,
                                             BufferSet& buffers) const
{
    auto status = ValidateGeneratedData(data, "retrieve");
    if (!status.Ok()) { return status; }

    buffers = BufferSet{};
    buffers.ownedBuffers.reserve(data.values.size());
    buffers.regions.reserve(data.values.size());
    buffers.entries.reserve(data.values.size());

    for (const auto& value : data.values) {
        buffers.ownedBuffers.emplace_back(value.size(), kRetrieveBufferInitialValue);
    }
    if (placement == PayloadBufferPlacement::ASCEND_DEVICE) {
        status = BuildDeviceBuffers(buffers);
        if (!status.Ok()) { return status; }
    }

    for (std::size_t index = 0; index < data.keys.size(); ++index) {
        auto region = MakeRegion(buffers, index, placement);
        if (placement != PayloadBufferPlacement::ASCEND_DEVICE) {
            buffers.regions.emplace_back(region);
        }
        buffers.entries.emplace_back(MakeKvBuffer(data.keys[index], region));
    }

    return Status::Success();
}

Status BufferAllocator::CopyDeviceBuffersToHost(BufferSet& buffers) const
{
    if (buffers.deviceBuffers.empty()) { return Status::Success(); }
    if (buffers.entries.size() != buffers.ownedBuffers.size()) {
        return Status::Error(kExitInvalidArgument,
                             "device payload entry/host buffer count mismatch");
    }

    for (std::size_t index = 0; index < buffers.ownedBuffers.size(); ++index) {
        auto& hostBuffer = buffers.ownedBuffers[index];
        if (hostBuffer.empty()) { continue; }
        auto ret = aclrtMemcpy(hostBuffer.data(), hostBuffer.size(),
                               reinterpret_cast<void*>(buffers.entries[index].buffer.region.addr),
                               hostBuffer.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            return Status::Error(
                kExitInvalidArgument,
                "device payload device-to-host copy failed: index=" + std::to_string(index) +
                    " size=" + std::to_string(hostBuffer.size()) + " ret=" + std::to_string(ret));
        }
    }
    return Status::Success();
}

}  // namespace UC::KVTest
