/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <algorithm>
#include <cstdint>
#include <vector>

#include <acl/acl.h>
#include <gtest/gtest.h>

#include "ffts_transport.h"

namespace {
constexpr int32_t kDeviceId = 0;

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes)
    {
        if (aclrtMalloc(&ptr_, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { ptr_ = nullptr; }
    }

    ~DeviceBuffer()
    {
        if (ptr_ != nullptr) { (void)aclrtFree(ptr_); }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* Get() const { return ptr_; }

private:
    void* ptr_{nullptr};
};

// Test fixture for ACL-based transport tests.
// It initializes ACL and sets the target device before each test,
// then resets the device and finalizes ACL after each test.
class FftsTransportTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        auto ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) { GTEST_SKIP() << "aclInit failed: " << ret; }
        aclInited_ = true;

        ret = aclrtSetDevice(kDeviceId);
        if (ret != ACL_SUCCESS) { GTEST_SKIP() << "aclrtSetDevice failed: " << ret; }
        deviceSet_ = true;
    }

    void TearDown() override
    {
        if (deviceSet_) { (void)aclrtResetDevice(kDeviceId); }
        if (aclInited_) { (void)aclFinalize(); }
    }

private:
    bool aclInited_{false};
    bool deviceSet_{false};
};

void FillPattern(std::vector<uint32_t>& data)
{
    for (size_t i = 0; i < data.size(); ++i) { data[i] = static_cast<uint32_t>(i ^ 0x5A5A5A5AU); }
}

void FillPipelinePattern(std::vector<uint32_t>& data)
{
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint32_t>((i * 1103515245ULL + 12345ULL) & 0xFFFFFFFFU);
    }
}

void* Offset(void* ptr, size_t bytes)
{
    return static_cast<void*>(static_cast<uint8_t*>(ptr) + bytes);
}

std::vector<UC::Transport::Ffts::CopyDesc> BuildChunkCopies(void* dst, const void* src, size_t chunkBytes,
                                                            size_t chunks)
{
    std::vector<UC::Transport::Ffts::CopyDesc> copies;
    copies.reserve(chunks);
    for (size_t i = 0; i < chunks; ++i) {
        const auto offset = chunkBytes * i;
        copies.push_back({Offset(dst, offset), Offset(const_cast<void*>(src), offset), chunkBytes});
    }
    return copies;
}
}  // namespace

TEST(FftsTransportInvalidTest, CopyBeforeSetupFails)
{
    UC::Transport::Ffts::FftsTransport transport;
    EXPECT_TRUE(transport.CopyAsync(reinterpret_cast<void*>(0x1), reinterpret_cast<void*>(0x2), 1).Failure());
}

TEST_F(FftsTransportTest, SingleIoCopyMatchesSource)
{
    constexpr size_t bytes = 4096;
    std::vector<uint32_t> src(bytes / sizeof(uint32_t));
    std::vector<uint32_t> dst(src.size(), 0);
    FillPattern(src);

    DeviceBuffer devSrc(bytes);
    DeviceBuffer devDst(bytes);
    ASSERT_NE(devSrc.Get(), nullptr);
    ASSERT_NE(devDst.Get(), nullptr);

    ASSERT_EQ(aclrtMemcpy(devSrc.Get(), bytes, src.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(devDst.Get(), bytes, 0, bytes), ACL_SUCCESS);

    UC::Transport::Ffts::FftsTransport transport;
    ASSERT_EQ(transport.Setup(kDeviceId), UC::Status::OK());
    ASSERT_EQ(transport.CopyAsync(devDst.Get(), devSrc.Get(), bytes), UC::Status::OK());
    ASSERT_EQ(transport.Synchronize(), UC::Status::OK());

    ASSERT_EQ(aclrtMemcpy(dst.data(), bytes, devDst.Get(), bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(dst, src);
}

TEST_F(FftsTransportTest, SubmitCopiesMultipleDeviceRanges)
{
    constexpr size_t chunkBytes = 1024;
    constexpr size_t chunks = 4;
    constexpr size_t totalBytes = chunkBytes * chunks;
    std::vector<uint32_t> src(totalBytes / sizeof(uint32_t));
    std::vector<uint32_t> dst(src.size(), 0);
    FillPattern(src);

    DeviceBuffer devSrc(totalBytes);
    DeviceBuffer devDst(totalBytes);
    ASSERT_NE(devSrc.Get(), nullptr);
    ASSERT_NE(devDst.Get(), nullptr);

    ASSERT_EQ(aclrtMemcpy(devSrc.Get(), totalBytes, src.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(devDst.Get(), totalBytes, 0, totalBytes), ACL_SUCCESS);

    auto copies = BuildChunkCopies(devDst.Get(), devSrc.Get(), chunkBytes, chunks);

    UC::Transport::Ffts::FftsTransport transport;
    ASSERT_EQ(transport.Setup(kDeviceId), UC::Status::OK());
    ASSERT_EQ(transport.Submit(copies), UC::Status::OK());
    ASSERT_EQ(transport.Synchronize(), UC::Status::OK());

    ASSERT_EQ(aclrtMemcpy(dst.data(), totalBytes, devDst.Get(), totalBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(dst, src);
}

TEST_F(FftsTransportTest, SubmitSplitsLargeBatchAcrossReadyContextLimit)
{
    constexpr size_t chunkBytes = 64;
    constexpr size_t chunks = 130;
    constexpr size_t totalBytes = chunkBytes * chunks;
    std::vector<uint32_t> src(totalBytes / sizeof(uint32_t));
    std::vector<uint32_t> dst(src.size(), 0);
    FillPattern(src);

    DeviceBuffer devSrc(totalBytes);
    DeviceBuffer devDst(totalBytes);
    ASSERT_NE(devSrc.Get(), nullptr);
    ASSERT_NE(devDst.Get(), nullptr);

    ASSERT_EQ(aclrtMemcpy(devSrc.Get(), totalBytes, src.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(devDst.Get(), totalBytes, 0, totalBytes), ACL_SUCCESS);

    UC::Transport::Ffts::FftsTransport transport;
    ASSERT_EQ(transport.Setup(kDeviceId), UC::Status::OK());
    ASSERT_EQ(transport.Submit(BuildChunkCopies(devDst.Get(), devSrc.Get(), chunkBytes, chunks)), UC::Status::OK());
    ASSERT_EQ(transport.Synchronize(), UC::Status::OK());

    ASSERT_EQ(aclrtMemcpy(dst.data(), totalBytes, devDst.Get(), totalBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(dst, src);
}

TEST_F(FftsTransportTest, TwoStagePipelineCopyReachesDestination)
{
    constexpr size_t chunkBytes = 1024;
    constexpr size_t chunks = 8;
    constexpr size_t totalBytes = chunkBytes * chunks;
    std::vector<uint32_t> src(totalBytes / sizeof(uint32_t));
    std::vector<uint32_t> dst(src.size(), 0);
    FillPipelinePattern(src);

    DeviceBuffer devSrc(totalBytes);
    DeviceBuffer devMid(totalBytes);
    DeviceBuffer devDst(totalBytes);
    ASSERT_NE(devSrc.Get(), nullptr);
    ASSERT_NE(devMid.Get(), nullptr);
    ASSERT_NE(devDst.Get(), nullptr);

    ASSERT_EQ(aclrtMemcpy(devSrc.Get(), totalBytes, src.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(devMid.Get(), totalBytes, 0, totalBytes), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(devDst.Get(), totalBytes, 0, totalBytes), ACL_SUCCESS);

    UC::Transport::Ffts::FftsTransport transport;
    ASSERT_EQ(transport.Setup(kDeviceId), UC::Status::OK());
    ASSERT_EQ(transport.Submit(BuildChunkCopies(devMid.Get(), devSrc.Get(), chunkBytes, chunks)), UC::Status::OK());
    ASSERT_EQ(transport.Submit(BuildChunkCopies(devDst.Get(), devMid.Get(), chunkBytes, chunks)), UC::Status::OK());
    ASSERT_EQ(transport.Synchronize(), UC::Status::OK());

    ASSERT_EQ(aclrtMemcpy(dst.data(), totalBytes, devDst.Get(), totalBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(dst, src);
}

TEST_F(FftsTransportTest, InvalidAndNoOpCopies)
{
    UC::Transport::Ffts::FftsTransport transport;
    ASSERT_EQ(transport.Setup(kDeviceId), UC::Status::OK());

    EXPECT_EQ(transport.Submit(nullptr, 0), UC::Status::OK());
    EXPECT_TRUE(transport.Submit(nullptr, 1).Failure());
    EXPECT_EQ(transport.CopyAsync(nullptr, nullptr, 0), UC::Status::OK());
    EXPECT_TRUE(transport.CopyAsync(nullptr, reinterpret_cast<void*>(0x1), 1).Failure());
}
