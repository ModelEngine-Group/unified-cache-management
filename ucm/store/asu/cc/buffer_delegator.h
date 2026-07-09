#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include "status/status.h"

namespace UC::AsuStore {

class BufferPool;
class Buffer;

class BufferMgr {
public:
    struct Allocator {
        std::function<void*(std::size_t)> allocate;
        std::function<void(void*)> free;
    };

    struct Config {
        std::size_t alignmentBytes{0};
        std::size_t alignedSize{0};
        std::size_t poolSizeBytes{0};
        Allocator allocator;
    };

    BufferMgr() = default;
    ~BufferMgr() = default;

    BufferMgr(const BufferMgr&) = delete;
    BufferMgr& operator=(const BufferMgr&) = delete;

    Status Init(Config config);
    Expected<Buffer> Acquire();

    bool IsInitialized() const noexcept { return initialized_; }
    std::size_t AlignmentBytes() const noexcept { return alignmentBytes_; }
    std::size_t AlignedSize() const noexcept { return alignedSize_; }
    std::size_t PoolSizeBytes() const noexcept { return poolSizeBytes_; }
    std::size_t AvailableSlots() const;

    static Allocator MakeAscendDeviceAllocator();

private:
    bool initialized_{false};
    std::size_t alignmentBytes_{0};
    std::size_t alignedSize_{0};
    std::size_t poolSizeBytes_{0};
    std::shared_ptr<BufferPool> pool_;
};

class BufferPool : public std::enable_shared_from_this<BufferPool> {
public:
    BufferPool(std::size_t slotSize, std::size_t slotCount, void* baseAddr,
               BufferMgr::Allocator allocator);
    ~BufferPool();

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    Expected<Buffer> Acquire();
    void Release(std::uint32_t slotId) noexcept;
    std::size_t AvailableSlots() const;
    std::size_t SlotCount() const noexcept { return slotCount_; }

private:
    std::size_t slotSize_{0};
    std::size_t slotCount_{0};
    void* baseAddr_{nullptr};
    BufferMgr::Allocator allocator_;

    mutable std::mutex mu_;
    std::vector<bool> inUse_;
    std::size_t availableSlots_{0};
};

class Buffer {
public:
    Buffer() = default;
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    explicit operator bool() const noexcept { return IsValid(); }
    bool IsValid() const noexcept { return pool_ != nullptr; }

    std::uint64_t DeviceAddr() const noexcept { return deviceAddr_; }
    std::size_t Capacity() const noexcept { return capacity_; }

    void Reset() noexcept;

private:
    Buffer(std::shared_ptr<BufferPool> pool, std::uint32_t slotId, std::uint64_t deviceAddr,
           std::size_t capacity);

    std::shared_ptr<BufferPool> pool_;
    std::uint32_t slotId_{UINT32_MAX};
    std::uint64_t deviceAddr_{0};
    std::size_t capacity_{0};

    friend class BufferPool;
};

}  // namespace UC::AsuStore
