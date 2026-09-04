#include "buffer_allocator.h"
#include <limits>
#include "key_value_generator.h"
#include "trans/device.h"

namespace UC::KVTest {

namespace {

constexpr int kExitInvalidArgument = 1;

constexpr std::uint8_t kRetrieveBufferInitialValue = 0xA5;
constexpr std::size_t kDeviceBufferAlignment = UC::ASU::kAsuAlignmentBytes;
constexpr std::size_t kDeviceMrRegisterAlignment = 2ULL * 1024ULL * 1024ULL;

Trans::Stream* GetThreadStream()
{
    thread_local std::unique_ptr<Trans::Stream> stream;
    if (!stream) { stream = Trans::Device{}.MakeStream(); }
    return stream.get();
}

Status BuildDeviceBuffers(BufferSet& buffers, DeviceAllocationPolicy allocationPolicy,
                          std::int32_t logicalDeviceId)
{
    std::size_t totalSize = 0;
    buffers.deviceBufferOffsets.clear();
    buffers.deviceBufferOffsets.reserve(buffers.ownedBuffers.size());
    for (const auto& hostBuffer : buffers.ownedBuffers) {
        const auto offset = AlignUp(totalSize, DeviceBufferAlignment());
        if (offset < totalSize ||
            hostBuffer.size() > std::numeric_limits<std::size_t>::max() - offset) {
            return Status::Error(kExitInvalidArgument, "device payload buffer size overflow");
        }
        buffers.deviceBufferOffsets.emplace_back(offset);
        totalSize = offset + hostBuffer.size();
    }

    if (totalSize == 0) { return Status::Success(); }

    const auto registerAlignment = DeviceAllocationAlignment(allocationPolicy);
    const auto registerSize = AlignUp(totalSize, registerAlignment);
    if (registerSize < totalSize) {
        return Status::Error(kExitInvalidArgument, "device payload register size overflow");
    }

    std::shared_ptr<void> deviceBuffer;
    auto status = AllocateDeviceBuffer(registerSize, allocationPolicy, deviceBuffer);
    if (!status.Ok()) { return status; }
    const auto baseAddr = reinterpret_cast<std::uintptr_t>(deviceBuffer.get());
    for (std::size_t index = 0; index < buffers.ownedBuffers.size(); ++index) {
        const auto deviceAddr = baseAddr + buffers.deviceBufferOffsets[index];
        status = CopyHostToDevice(buffers.ownedBuffers[index], deviceAddr, "");
        if (!status.Ok()) { return status; }
    }

    buffers.deviceBuffers.emplace_back(std::move(deviceBuffer));
    buffers.regions.emplace_back(MakeDeviceRegion(baseAddr, registerSize, logicalDeviceId));
    buffers.entryRegionIndexes.assign(buffers.ownedBuffers.size(), 0);
    return Status::Success();
}

UC::ASU::MemoryRegion MakeRegion(BufferSet& buffers, std::size_t index,
                                 PayloadBufferPlacement placement, std::int32_t logicalDeviceId)
{
    if (placement != PayloadBufferPlacement::HOST) {
        const auto baseAddr =
            buffers.deviceBuffers.empty()
                ? 0
                : reinterpret_cast<std::uintptr_t>(buffers.deviceBuffers.front().get());
        return MakeDeviceRegion(baseAddr + buffers.deviceBufferOffsets[index],
                                buffers.ownedBuffers[index].size(), logicalDeviceId);
    }
    return MakeHostRegion(buffers.ownedBuffers[index]);
}

}  // namespace

std::size_t DeviceBufferAlignment() { return kDeviceBufferAlignment; }

std::size_t DeviceMrRegisterAlignment() { return kDeviceMrRegisterAlignment; }

std::size_t DeviceAllocationAlignment(DeviceAllocationPolicy allocationPolicy)
{
    return allocationPolicy == DeviceAllocationPolicy::AIV_REGISTERABLE
               ? DeviceMrRegisterAlignment()
               : DeviceBufferAlignment();
}

std::size_t AlignUp(std::size_t value, std::size_t alignment)
{
    if (alignment == 0) { return value; }
    const auto remainder = value % alignment;
    if (remainder == 0) { return value; }
    return value + alignment - remainder;
}

Status AllocateDeviceBuffer(std::size_t size, DeviceAllocationPolicy allocationPolicy,
                            std::shared_ptr<void>& deviceBuffer)
{
    if (size == 0) {
        deviceBuffer.reset();
        return Status::Success();
    }

    const auto alignment = DeviceAllocationAlignment(allocationPolicy);
    if (size > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        return Status::Error(kExitInvalidArgument, "device payload allocation size overflow");
    }
    const auto allocationSize = size + alignment - 1;

    auto buffer = Trans::Device{}.MakeBuffer();
    if (!buffer) {
        return Status::Error(kExitInvalidArgument, "device payload buffer creation failed");
    }
    auto owner = buffer->MakeDeviceBuffer(allocationSize);
    if (!owner) {
        return Status::Error(kExitInvalidArgument, "device payload MakeDeviceBuffer failed: size=" +
                                                       std::to_string(allocationSize));
    }
    const auto baseAddr = AlignUp(reinterpret_cast<std::uintptr_t>(owner.get()), alignment);
    deviceBuffer = std::shared_ptr<void>(std::move(owner), reinterpret_cast<void*>(baseAddr));
    return Status::Success();
}

Status CopyHostToDevice(const std::vector<std::uint8_t>& hostBuffer, std::uintptr_t deviceAddr,
                        const std::string& context)
{
    if (hostBuffer.empty()) { return Status::Success(); }
    auto* stream = GetThreadStream();
    if (stream == nullptr) {
        return Status::Error(kExitInvalidArgument, "device payload stream creation failed");
    }
    const auto status =
        stream->HostToDevice(const_cast<void*>(static_cast<const void*>(hostBuffer.data())),
                             reinterpret_cast<void*>(deviceAddr), hostBuffer.size());
    if (!status.Success()) {
        const auto contextText = context.empty() ? "" : " " + context;
        return Status::Error(kExitInvalidArgument,
                             "device payload" + contextText + " host-to-device copy failed: size=" +
                                 std::to_string(hostBuffer.size()) + " " + status.ToString());
    }
    return Status::Success();
}

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

UC::ASU::MemoryRegion MakeDeviceRegion(std::uint64_t addr, std::size_t size,
                                       std::int32_t logicalDeviceId)
{
    UC::ASU::MemoryRegion region;
    region.memoryType = UC::ASU::MemoryType::DEVICE;
    region.addr = addr;
    region.size = size;
    region.deviceId = logicalDeviceId;
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

Status BufferAllocator::BuildStoreBuffers(const GeneratedData& data,
                                          PayloadBufferPlacement placement,
                                          DeviceAllocationPolicy allocationPolicy,
                                          std::int32_t logicalDeviceId, BufferSet& buffers) const
{
    auto status = ValidateGeneratedData(data, "store");
    if (!status.Ok()) { return status; }

    buffers = BufferSet{};
    buffers.ownedBuffers.reserve(data.values.size());
    buffers.regions.reserve(data.values.size());
    buffers.entries.reserve(data.values.size());

    for (const auto& value : data.values) { buffers.ownedBuffers.emplace_back(value); }
    if (placement != PayloadBufferPlacement::HOST) {
        status = BuildDeviceBuffers(buffers, allocationPolicy, logicalDeviceId);
        if (!status.Ok()) { return status; }
    }

    for (std::size_t index = 0; index < data.keys.size(); ++index) {
        auto region = MakeRegion(buffers, index, placement, logicalDeviceId);
        if (placement == PayloadBufferPlacement::HOST) { buffers.regions.emplace_back(region); }
        buffers.entries.emplace_back(MakeKvBuffer(data.keys[index], region));
    }

    return Status::Success();
}

Status BufferAllocator::BuildRetrieveBuffers(const GeneratedData& data,
                                             PayloadBufferPlacement placement,
                                             DeviceAllocationPolicy allocationPolicy,
                                             std::int32_t logicalDeviceId, BufferSet& buffers) const
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
    if (placement != PayloadBufferPlacement::HOST) {
        status = BuildDeviceBuffers(buffers, allocationPolicy, logicalDeviceId);
        if (!status.Ok()) { return status; }
    }

    for (std::size_t index = 0; index < data.keys.size(); ++index) {
        auto region = MakeRegion(buffers, index, placement, logicalDeviceId);
        if (placement == PayloadBufferPlacement::HOST) { buffers.regions.emplace_back(region); }
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

    auto* stream = GetThreadStream();
    if (stream == nullptr) {
        return Status::Error(kExitInvalidArgument, "device payload stream creation failed");
    }

    for (std::size_t index = 0; index < buffers.ownedBuffers.size(); ++index) {
        auto& hostBuffer = buffers.ownedBuffers[index];
        if (hostBuffer.empty()) { continue; }
        const auto status =
            stream->DeviceToHost(reinterpret_cast<void*>(buffers.entries[index].buffer.region.addr),
                                 hostBuffer.data(), hostBuffer.size());
        if (!status.Success()) {
            return Status::Error(
                kExitInvalidArgument,
                "device payload device-to-host copy failed: index=" + std::to_string(index) +
                    " size=" + std::to_string(hostBuffer.size()) + " " + status.ToString());
        }
    }
    return Status::Success();
}

}  // namespace UC::KVTest
