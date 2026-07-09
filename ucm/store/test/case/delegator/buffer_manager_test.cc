#include "buffer_manager.h"
#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

namespace {

constexpr std::size_t kAlignmentBytes = 16 * 1024;
constexpr std::size_t kAlignedSize = 163840;

class BufferMgrTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        const auto ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
            FAIL() << "aclInit failed: " << ret;
        }
        ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    }

    static void TearDownTestSuite()
    {
        (void)aclrtResetDevice(0);
        (void)aclFinalize();
    }
};

UC::BufferDelegator::BufferMgr::Config MakeConfig(std::size_t poolSizeBytes = 2 * kAlignedSize)
{
    UC::BufferDelegator::BufferMgr::Config config;
    config.alignmentBytes = kAlignmentBytes;
    config.alignedSize = kAlignedSize;
    config.poolSizeBytes = poolSizeBytes;
    config.allocator = UC::BufferDelegator::BufferMgr::MakeAscendDeviceAllocator();
    return config;
}

}  // namespace

TEST_F(BufferMgrTest, InitCreatesOneFixedPool)
{
    {
        UC::BufferDelegator::BufferMgr mgr;

        auto status = mgr.Init(MakeConfig(2 * kAlignedSize + 1));
        ASSERT_TRUE(status.Success()) << status.ToString();
        EXPECT_TRUE(mgr.IsInitialized());
        EXPECT_EQ(mgr.AlignmentBytes(), kAlignmentBytes);
        EXPECT_EQ(mgr.AlignedSize(), kAlignedSize);
        EXPECT_EQ(mgr.PoolSizeBytes(), 2 * kAlignedSize);
        EXPECT_EQ(mgr.AvailableSlots(), std::size_t{2});

        auto first = mgr.Acquire();
        ASSERT_TRUE(first.HasValue()) << first.Error().ToString();
        EXPECT_EQ(first.Value().Capacity(), kAlignedSize);
        EXPECT_NE(first.Value().DeviceAddr(), std::uint64_t{0});
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
}

TEST_F(BufferMgrTest, BufferReturnsSlotOnResetAndDestruction)
{
    UC::BufferDelegator::BufferMgr mgr;
    ASSERT_TRUE(mgr.Init(MakeConfig()).Success());
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
}

TEST_F(BufferMgrTest, MoveTransfersBufferOwnership)
{
    UC::BufferDelegator::BufferMgr mgr;
    ASSERT_TRUE(mgr.Init(MakeConfig()).Success());

    auto buffer = mgr.Acquire();
    ASSERT_TRUE(buffer.HasValue()) << buffer.Error().ToString();
    auto moved = std::move(buffer.Value());
    EXPECT_FALSE(buffer.Value().IsValid());
    EXPECT_TRUE(moved.IsValid());
    EXPECT_EQ(mgr.AvailableSlots(), std::size_t{1});

    moved.Reset();
    EXPECT_EQ(mgr.AvailableSlots(), std::size_t{2});
}

TEST_F(BufferMgrTest, RejectsInvalidConfigAndAcquireBeforeInit)
{
    UC::BufferDelegator::BufferMgr mgr;

    auto beforeInit = mgr.Acquire();
    ASSERT_FALSE(beforeInit.HasValue());
    EXPECT_EQ(beforeInit.Error(), UC::Status::InvalidParam());

    auto config = MakeConfig();
    config.alignmentBytes = 0;
    EXPECT_TRUE(mgr.Init(config).Failure());

    config = MakeConfig();
    config.alignedSize = 0;
    EXPECT_TRUE(mgr.Init(config).Failure());

    config = MakeConfig();
    config.alignedSize = 1;
    EXPECT_TRUE(mgr.Init(config).Failure());

    config = MakeConfig();
    config.poolSizeBytes = 0;
    EXPECT_TRUE(mgr.Init(config).Failure());

    config = MakeConfig(kAlignedSize - 1);
    EXPECT_TRUE(mgr.Init(config).Failure());

    config = MakeConfig();
    config.allocator.allocate = nullptr;
    EXPECT_TRUE(mgr.Init(config).Failure());

    {
        UC::BufferDelegator::BufferMgr validMgr;
        ASSERT_TRUE(validMgr.Init(MakeConfig()).Success());
        EXPECT_TRUE(validMgr.Init(MakeConfig()).Failure());
    }
}
