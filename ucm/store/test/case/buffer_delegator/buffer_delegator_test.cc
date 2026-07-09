#include "buffer_delegator.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <unordered_map>

namespace {

constexpr std::size_t kAlignmentBytes = 16 * 1024;
constexpr std::size_t kAlignedSize = 163840;

class FakeDeviceAllocator {
public:
    UC::BufferDelegator::BufferMgr::Allocator MakeAllocator()
    {
        return UC::BufferDelegator::BufferMgr::Allocator{
            [this](std::size_t size) { return Allocate(size); },
            [this](void* ptr) { Free(ptr); },
        };
    }

    std::size_t LiveAllocations() const { return allocations_.size(); }
    void* LiveAllocation() const
    {
        if (allocations_.empty()) { return nullptr; }
        return allocations_.begin()->first;
    }

private:
    void* Allocate(std::size_t size)
    {
        void* base = std::malloc(size + 1);
        if (base == nullptr) { return nullptr; }
        auto* raw = static_cast<std::byte*>(base) + 1;
        allocations_[raw] = base;
        return raw;
    }

    void Free(void* ptr)
    {
        auto iter = allocations_.find(ptr);
        if (iter == allocations_.end()) { return; }
        std::free(iter->second);
        allocations_.erase(iter);
    }

    std::unordered_map<void*, void*> allocations_;
};

UC::BufferDelegator::BufferMgr::Config MakeConfig(
    FakeDeviceAllocator& allocator, std::size_t poolSizeBytes = 2 * kAlignedSize)
{
    UC::BufferDelegator::BufferMgr::Config config;
    config.alignmentBytes = kAlignmentBytes;
    config.alignedSize = kAlignedSize;
    config.poolSizeBytes = poolSizeBytes;
    config.allocator = allocator.MakeAllocator();
    return config;
}

}  // namespace

TEST(BufferMgrTest, InitCreatesOneFixedPool)
{
    FakeDeviceAllocator allocator;
    {
        UC::BufferDelegator::BufferMgr mgr;

        auto status = mgr.Init(MakeConfig(allocator, 2 * kAlignedSize + 1));
        ASSERT_TRUE(status.Success()) << status.ToString();
        EXPECT_TRUE(mgr.IsInitialized());
        EXPECT_EQ(mgr.AlignmentBytes(), kAlignmentBytes);
        EXPECT_EQ(mgr.AlignedSize(), kAlignedSize);
        EXPECT_EQ(mgr.PoolSizeBytes(), 2 * kAlignedSize);
        EXPECT_EQ(mgr.AvailableSlots(), std::size_t{2});
        EXPECT_EQ(allocator.LiveAllocations(), std::size_t{1});

        auto first = mgr.Acquire();
        ASSERT_TRUE(first.HasValue()) << first.Error().ToString();
        EXPECT_EQ(first.Value().Capacity(), kAlignedSize);
        EXPECT_EQ(first.Value().DeviceAddr(),
                  reinterpret_cast<std::uint64_t>(allocator.LiveAllocation()));
        EXPECT_EQ(mgr.AvailableSlots(), std::size_t{1});

        auto second = mgr.Acquire();
        ASSERT_TRUE(second.HasValue()) << second.Error().ToString();
        EXPECT_EQ(second.Value().Capacity(), kAlignedSize);
        EXPECT_EQ(second.Value().DeviceAddr(), first.Value().DeviceAddr() + kAlignedSize);
        EXPECT_EQ(mgr.AvailableSlots(), std::size_t{0});

        auto third = mgr.Acquire();
        ASSERT_FALSE(third.HasValue());
        EXPECT_EQ(third.Error(), UC::Status::Retry());
    }

    EXPECT_EQ(allocator.LiveAllocations(), std::size_t{0});
}

TEST(BufferMgrTest, BufferReturnsSlotOnResetAndDestruction)
{
    FakeDeviceAllocator allocator;
    UC::BufferDelegator::BufferMgr mgr;
    ASSERT_TRUE(mgr.Init(MakeConfig(allocator)).Success());
    EXPECT_EQ(mgr.AvailableSlots(), std::size_t{2});

    {
        auto buffer = mgr.Acquire();
        ASSERT_TRUE(buffer.HasValue()) << buffer.Error().ToString();
        EXPECT_EQ(mgr.AvailableSlots(), std::size_t{1});
        buffer.Value().Reset();
        EXPECT_EQ(mgr.AvailableSlots(), std::size_t{2});
        buffer.Value().Reset();
        EXPECT_EQ(mgr.AvailableSlots(), std::size_t{2});
    }

    {
        auto buffer = mgr.Acquire();
        ASSERT_TRUE(buffer.HasValue()) << buffer.Error().ToString();
        EXPECT_EQ(mgr.AvailableSlots(), std::size_t{1});
    }
    EXPECT_EQ(mgr.AvailableSlots(), std::size_t{2});
    EXPECT_EQ(allocator.LiveAllocations(), std::size_t{1});
}

TEST(BufferMgrTest, MoveTransfersBufferOwnership)
{
    FakeDeviceAllocator allocator;
    UC::BufferDelegator::BufferMgr mgr;
    ASSERT_TRUE(mgr.Init(MakeConfig(allocator)).Success());

    auto buffer = mgr.Acquire();
    ASSERT_TRUE(buffer.HasValue()) << buffer.Error().ToString();
    auto moved = std::move(buffer.Value());
    EXPECT_FALSE(buffer.Value().IsValid());
    EXPECT_TRUE(moved.IsValid());
    EXPECT_EQ(mgr.AvailableSlots(), std::size_t{1});

    moved.Reset();
    EXPECT_EQ(mgr.AvailableSlots(), std::size_t{2});
}

TEST(BufferMgrTest, RejectsInvalidConfigAndAcquireBeforeInit)
{
    FakeDeviceAllocator allocator;
    UC::BufferDelegator::BufferMgr mgr;

    auto beforeInit = mgr.Acquire();
    ASSERT_FALSE(beforeInit.HasValue());
    EXPECT_EQ(beforeInit.Error(), UC::Status::InvalidParam());

    auto config = MakeConfig(allocator);
    config.alignmentBytes = 0;
    EXPECT_TRUE(mgr.Init(config).Failure());

    config = MakeConfig(allocator);
    config.alignedSize = 0;
    EXPECT_TRUE(mgr.Init(config).Failure());

    config = MakeConfig(allocator);
    config.alignedSize = 1;
    EXPECT_TRUE(mgr.Init(config).Failure());

    config = MakeConfig(allocator);
    config.poolSizeBytes = 0;
    EXPECT_TRUE(mgr.Init(config).Failure());

    config = MakeConfig(allocator, kAlignedSize - 1);
    EXPECT_TRUE(mgr.Init(config).Failure());

    config = MakeConfig(allocator);
    config.allocator.allocate = nullptr;
    EXPECT_TRUE(mgr.Init(config).Failure());

    {
        UC::BufferDelegator::BufferMgr validMgr;
        ASSERT_TRUE(validMgr.Init(MakeConfig(allocator)).Success());
        EXPECT_TRUE(validMgr.Init(MakeConfig(allocator)).Failure());
        EXPECT_EQ(allocator.LiveAllocations(), std::size_t{1});
    }
    EXPECT_EQ(allocator.LiveAllocations(), std::size_t{0});
}
