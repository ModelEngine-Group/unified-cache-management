/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "ipc/mem_fd.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>

namespace UC {
namespace {

TEST(MemFdTest, PopulateRejectsUnmappedMemory)
{
    MemFd mem;
    EXPECT_EQ(Status::InvalidParam(), mem.PopulateWrite());
}

TEST(MemFdTest, PopulateMakesSharedMemoryWritable)
{
    const auto pageSize = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    MemFd owner;
    ASSERT_TRUE(owner.Create("ucm_populate_test", pageSize * 2).Success());
    // The running kernel, rather than the build headers, determines support.
    const auto result = owner.PopulateWrite();
    if (result.Failure() && result.ToString().find("errno=22 ") != std::string::npos) {
        GTEST_SKIP() << result.ToString();
    }
    ASSERT_TRUE(result.Success()) << result.ToString();
    auto* bytes = static_cast<uint8_t*>(owner.Addr());
    bytes[0] = 0x5a;
    bytes[pageSize] = 0xa5;
    MemFd peer;
    ASSERT_TRUE(peer.Adopt(::dup(owner.Fd()), pageSize * 2).Success());
    EXPECT_EQ(0x5a, static_cast<uint8_t*>(peer.Addr())[0]);
    EXPECT_EQ(0xa5, static_cast<uint8_t*>(peer.Addr())[pageSize]);
}

TEST(MemFdTest, PopulateReportsMissingBackingWithoutSigbus)
{
    const auto pageSize = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    MemFd owner;
    ASSERT_TRUE(owner.Create("ucm_populate_short_file", pageSize).Success());
    MemFd peer;
    // Accessing the second page directly would deliver SIGBUS. A prefault must
    // return an error instead, including on kernels with no populate support.
    ASSERT_TRUE(peer.Adopt(::dup(owner.Fd()), pageSize * 2).Success());
    const auto result = peer.PopulateWrite();
    EXPECT_TRUE(result.Failure());
    EXPECT_NE(std::string::npos, result.ToString().find("MADV_POPULATE_WRITE"));
    // The process and the original mapping remain usable after the failure.
    static_cast<uint8_t*>(owner.Addr())[0] = 0x7c;
    EXPECT_EQ(0x7c, static_cast<uint8_t*>(peer.Addr())[0]);
}

}  // namespace
}  // namespace UC
