#include "buffer_delegator.h"
#include <acl/acl.h>
#include <limits>
#include <utility>

namespace UC::BufferDelegator {
namespace {

bool IsAligned(std::size_t value, std::size_t alignment)
{
    return value % alignment == 0;
}

}  // namespace

BufferPool::BufferPool(std::size_t slotSize, std::size_t slotCount, void* baseAddr,
                       BufferMgr::Allocator allocator)
    : slotSize_(slotSize),
      slotCount_(slotCount),
      baseAddr_(baseAddr),
      allocator_(std::move(allocator)),
      inUse_(slotCount, false),
      availableSlots_(slotCount)
{
}

BufferPool::~BufferPool()
{
    if (baseAddr_ != nullptr && allocator_.free) { allocator_.free(baseAddr_); }
}

Expected<Buffer> BufferPool::Acquire()
{
    std::lock_guard<std::mutex> lock(mu_);
    for (std::size_t index = 0; index < inUse_.size(); ++index) {
        if (inUse_[index]) { continue; }
        inUse_[index] = true;
        --availableSlots_;
        const auto deviceAddr = reinterpret_cast<std::uintptr_t>(baseAddr_) + index * slotSize_;
        return Buffer{shared_from_this(), static_cast<std::uint32_t>(index),
                      static_cast<std::uint64_t>(deviceAddr), slotSize_};
    }
    return Status::Retry();
}

void BufferPool::Release(std::uint32_t slotId) noexcept
{
    std::lock_guard<std::mutex> lock(mu_);
    if (slotId >= slotCount_ || !inUse_[slotId]) { return; }
    inUse_[slotId] = false;
    ++availableSlots_;
}

std::size_t BufferPool::AvailableSlots() const
{
    std::lock_guard<std::mutex> lock(mu_);
    return availableSlots_;
}

Buffer::Buffer(std::shared_ptr<BufferPool> pool, std::uint32_t slotId,
               std::uint64_t deviceAddr, std::size_t capacity)
    : pool_(std::move(pool)),
      slotId_(slotId),
      deviceAddr_(deviceAddr),
      capacity_(capacity)
{
}

Buffer::~Buffer()
{
    Reset();
}

Buffer::Buffer(Buffer&& other) noexcept
{
    *this = std::move(other);
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
{
    if (this == &other) { return *this; }
    Reset();
    pool_ = std::move(other.pool_);
    slotId_ = other.slotId_;
    deviceAddr_ = other.deviceAddr_;
    capacity_ = other.capacity_;
    other.slotId_ = UINT32_MAX;
    other.deviceAddr_ = 0;
    other.capacity_ = 0;
    return *this;
}

void Buffer::Reset() noexcept
{
    if (!pool_) { return; }
    pool_->Release(slotId_);
    pool_.reset();
    slotId_ = UINT32_MAX;
    deviceAddr_ = 0;
    capacity_ = 0;
}

Status BufferMgr::Init(Config config)
{
    if (initialized_) {
        return Status::InvalidParam("buffer delegator buffer mgr already initialized");
    }
    if (config.alignmentBytes == 0) {
        return Status::InvalidParam("buffer delegator alignment must be non-zero");
    }
    if (config.alignedSize == 0) {
        return Status::InvalidParam("buffer delegator aligned size must be non-zero");
    }
    if (!IsAligned(config.alignedSize, config.alignmentBytes)) {
        return Status::InvalidParam("buffer delegator aligned size is not aligned");
    }
    if (config.poolSizeBytes == 0) {
        return Status::InvalidParam("buffer delegator pool size must be non-zero");
    }
    if (!config.allocator.allocate || !config.allocator.free) {
        return Status::InvalidParam("buffer delegator allocator is incomplete");
    }

    const auto slotCount = config.poolSizeBytes / config.alignedSize;
    if (slotCount == 0) {
        return Status::InvalidParam("buffer delegator pool cannot hold one buffer");
    }
    if (slotCount > std::numeric_limits<std::uint32_t>::max()) {
        return Status::InvalidParam("buffer delegator pool slot count is too large");
    }

    void* baseAddr = config.allocator.allocate(config.poolSizeBytes);
    if (baseAddr == nullptr) { return Status::OutOfMemory(); }

    try {
        pool_ = std::make_shared<BufferPool>(config.alignedSize, slotCount, baseAddr,
                                             config.allocator);
    } catch (...) {
        config.allocator.free(baseAddr);
        return Status::OutOfMemory();
    }

    alignmentBytes_ = config.alignmentBytes;
    alignedSize_ = config.alignedSize;
    poolSizeBytes_ = slotCount * config.alignedSize;
    initialized_ = true;
    return Status::OK();
}

Expected<Buffer> BufferMgr::Acquire()
{
    if (!initialized_) {
        return Status::InvalidParam("buffer delegator buffer mgr not initialized");
    }
    return pool_->Acquire();
}

std::size_t BufferMgr::AvailableSlots() const
{
    if (!pool_) { return 0; }
    return pool_->AvailableSlots();
}

BufferMgr::Allocator BufferMgr::MakeAscendDeviceAllocator()
{
    return Allocator{
        [](std::size_t size) -> void* {
            void* device = nullptr;
            const auto ret = aclrtMalloc(&device, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
            return ret == ACL_SUCCESS ? device : nullptr;
        },
        [](void* ptr) {
            if (ptr != nullptr) { (void)aclrtFree(ptr); }
        },
    };
}

}  // namespace UC::BufferDelegator
