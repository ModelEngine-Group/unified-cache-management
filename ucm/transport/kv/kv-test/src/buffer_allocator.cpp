#include "kv_test/buffer_allocator.h"

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

}  // namespace

Status BufferAllocator::BuildStoreBuffers(const GeneratedData& data, BufferSet& buffers) const
{
    auto status = ValidateGeneratedData(data, "store");
    if (!status.Ok()) { return status; }

    buffers = BufferSet{};
    buffers.ownedBuffers.reserve(data.values.size());
    buffers.regions.reserve(data.values.size());
    buffers.entries.reserve(data.values.size());

    for (const auto& value : data.values) { buffers.ownedBuffers.emplace_back(value); }

    for (std::size_t index = 0; index < data.keys.size(); ++index) {
        auto region = MakeHostRegion(buffers.ownedBuffers[index]);
        buffers.regions.emplace_back(region);
        buffers.entries.emplace_back(MakeKvBuffer(data.keys[index], region));
    }

    return Status::Success();
}

Status BufferAllocator::BuildRetrieveBuffers(const GeneratedData& data, BufferSet& buffers) const
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

    for (std::size_t index = 0; index < data.keys.size(); ++index) {
        auto region = MakeHostRegion(buffers.ownedBuffers[index]);
        buffers.regions.emplace_back(region);
        buffers.entries.emplace_back(MakeKvBuffer(data.keys[index], region));
    }

    return Status::Success();
}

}  // namespace UC::KVTest
