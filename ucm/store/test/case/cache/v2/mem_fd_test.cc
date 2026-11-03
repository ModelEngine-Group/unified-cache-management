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

TEST(MemFdTest, PreallocateRejectsUnmappedMemory)
{
    MemFd mem;
    EXPECT_EQ(Status::InvalidParam(), mem.Preallocate());
}

TEST(MemFdTest, PreallocateMakesSharedMemoryWritable)
{
    const auto pageSize = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    MemFd owner;
    ASSERT_TRUE(owner.Create("ucm_preallocate_test", pageSize * 2).Success());
    const auto result = owner.Preallocate();
    ASSERT_TRUE(result.Success()) << result.ToString();
    auto* bytes = static_cast<uint8_t*>(owner.Addr());
    bytes[0] = 0x5a;
    bytes[pageSize] = 0xa5;
    MemFd peer;
    ASSERT_TRUE(peer.Adopt(::dup(owner.Fd()), pageSize * 2).Success());
    EXPECT_EQ(0x5a, static_cast<uint8_t*>(peer.Addr())[0]);
    EXPECT_EQ(0xa5, static_cast<uint8_t*>(peer.Addr())[pageSize]);
}

TEST(MemFdTest, PreallocateReportsFailureWithoutSigbus)
{
    const auto pageSize = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    const int fd = ::memfd_create("ucm_preallocate_short_file", MFD_ALLOW_SEALING | MFD_CLOEXEC);
    ASSERT_GE(fd, 0);
    MemFd owner;
    ASSERT_TRUE(owner.Adopt(fd, pageSize).Success());
    ASSERT_EQ(0, ::ftruncate(fd, static_cast<off_t>(pageSize)));
    ASSERT_EQ(0, ::fcntl(fd, F_ADD_SEALS, F_SEAL_GROW));
    MemFd peer;
    // Accessing the second page directly would deliver SIGBUS. Growing the
    // sealed file is forbidden, so preallocation must report a syscall error.
    ASSERT_TRUE(peer.Adopt(::dup(owner.Fd()), pageSize * 2).Success());
    const auto result = peer.Preallocate();
    EXPECT_TRUE(result.Failure());
    EXPECT_NE(std::string::npos, result.ToString().find("fallocate memfd failed"));
    // The process and the original mapping remain usable after the failure.
    static_cast<uint8_t*>(owner.Addr())[0] = 0x7c;
    EXPECT_EQ(0x7c, static_cast<uint8_t*>(peer.Addr())[0]);
}

}  // namespace
}  // namespace UC
