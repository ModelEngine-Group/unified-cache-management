#include "kv_test/buffer_allocator.h"
#include <acl/acl.h>

namespace UC::KVTest {

namespace {

constexpr int kExitInvalidArgument = 1;

constexpr std::uint8_t kRetrieveBufferInitialValue = 0xA5;

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

UC::ASU::MemoryRegion MakeDeviceRegion(const std::shared_ptr<void>& buffer, std::size_t size)
{
    UC::ASU::MemoryRegion region;
    region.memoryType = UC::ASU::MemoryType::ASCEND_DEVICE;
    region.addr = buffer ? reinterpret_cast<std::uint64_t>(buffer.get()) : 0;
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

Status MakeDeviceBuffer(const std::vector<std::uint8_t>& hostBuffer,
                        std::shared_ptr<void>& deviceBuffer)
{
    if (hostBuffer.empty()) {
        deviceBuffer.reset();
        return Status::Success();
    }

    void* ptr = nullptr;
    auto ret = aclrtMalloc(&ptr, hostBuffer.size(), ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    if (ret != ACL_SUCCESS) {
        return Status::Error(kExitInvalidArgument, "fake_backend aclrtMalloc failed: size=" +
                                                       std::to_string(hostBuffer.size()) +
                                                       " ret=" + std::to_string(ret));
    }
    deviceBuffer = std::shared_ptr<void>(ptr, aclrtFree);

    ret = aclrtMemcpy(deviceBuffer.get(), hostBuffer.size(), hostBuffer.data(), hostBuffer.size(),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        deviceBuffer.reset();
        return Status::Error(kExitInvalidArgument,
                             "fake_backend host-to-device copy failed: size=" +
                                 std::to_string(hostBuffer.size()) + " ret=" + std::to_string(ret));
    }
    return Status::Success();
}

Status BuildDeviceBuffers(BufferSet& buffers)
{
    buffers.deviceBuffers.reserve(buffers.ownedBuffers.size());
    for (const auto& hostBuffer : buffers.ownedBuffers) {
        std::shared_ptr<void> deviceBuffer;
        auto status = MakeDeviceBuffer(hostBuffer, deviceBuffer);
        if (!status.Ok()) { return status; }
        buffers.deviceBuffers.emplace_back(std::move(deviceBuffer));
    }
    return Status::Success();
}

UC::ASU::MemoryRegion MakeRegion(BufferSet& buffers, std::size_t index,
                                 PayloadBufferPlacement placement)
{
    if (placement == PayloadBufferPlacement::ASCEND_DEVICE) {
        return MakeDeviceRegion(buffers.deviceBuffers[index], buffers.ownedBuffers[index].size());
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
        buffers.regions.emplace_back(region);
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
        buffers.regions.emplace_back(region);
        buffers.entries.emplace_back(MakeKvBuffer(data.keys[index], region));
    }

    return Status::Success();
}

Status BufferAllocator::CopyDeviceBuffersToHost(BufferSet& buffers) const
{
    if (buffers.deviceBuffers.empty()) { return Status::Success(); }
    if (buffers.deviceBuffers.size() != buffers.ownedBuffers.size()) {
        return Status::Error(kExitInvalidArgument,
                             "fake_backend device/host buffer count mismatch");
    }

    for (std::size_t index = 0; index < buffers.ownedBuffers.size(); ++index) {
        auto& hostBuffer = buffers.ownedBuffers[index];
        if (hostBuffer.empty()) { continue; }
        auto ret =
            aclrtMemcpy(hostBuffer.data(), hostBuffer.size(), buffers.deviceBuffers[index].get(),
                        hostBuffer.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            return Status::Error(
                kExitInvalidArgument,
                "fake_backend device-to-host copy failed: index=" + std::to_string(index) +
                    " size=" + std::to_string(hostBuffer.size()) + " ret=" + std::to_string(ret));
        }
    }
    return Status::Success();
}

}  // namespace UC::KVTest
