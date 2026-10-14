/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "cache/v2/data_backend_memfd.h"
#include <cstdint>
#include <gtest/gtest.h>

namespace UC::Cache2 {

struct MemfdDataBackendTestAccess {
    static Status ResolveRegistrationChunkBytes(size_t rankStride, size_t slotSize, size_t pageSize,
                                                size_t maxBytes, size_t& chunkBytes)
    {
        return MemfdDataBackend::ResolveRegistrationChunkBytes(rankStride, slotSize, pageSize,
                                                               maxBytes, chunkBytes);
    }

    static void SeedRegistrations(MemfdDataBackend& backend, size_t rankStride, size_t chunkBytes,
                                  void* firstAlias, void* secondAlias)
    {
        backend.rankStride_ = rankStride;
        backend.registrationChunkBytes_ = chunkBytes;
        backend.segments_ = std::vector<MemfdDataBackend::Segment>(1);
        backend.segments_[0].registrations = {
            MemfdDataBackend::Registration{0,          chunkBytes,              firstAlias },
            MemfdDataBackend::Registration{chunkBytes, rankStride - chunkBytes, secondAlias},
        };
    }
};

namespace {

TEST(MemfdDataBackendTest, KeepsSmallSegmentInOneRegistration)
{
    size_t chunkBytes = 0;
    auto status = MemfdDataBackendTestAccess::ResolveRegistrationChunkBytes(
        /*rankStride=*/8 * 1024, /*slotSize=*/4 * 1024, /*pageSize=*/4 * 1024,
        /*maxBytes=*/16 * 1024, chunkBytes);

    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(8 * 1024, chunkBytes);
}

TEST(MemfdDataBackendTest, CapsRegistrationAtConfiguredMaximum)
{
    size_t chunkBytes = 0;
    auto status = MemfdDataBackendTestAccess::ResolveRegistrationChunkBytes(
        /*rankStride=*/32 * 1024, /*slotSize=*/4 * 1024, /*pageSize=*/4 * 1024,
        /*maxBytes=*/16 * 1024, chunkBytes);

    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(16 * 1024, chunkBytes);
}

TEST(MemfdDataBackendTest, AlignsChunkToSlotAndPageBoundaries)
{
    size_t chunkBytes = 0;
    auto status = MemfdDataBackendTestAccess::ResolveRegistrationChunkBytes(
        /*rankStride=*/48 * 1024, /*slotSize=*/6 * 1024, /*pageSize=*/4 * 1024,
        /*maxBytes=*/16 * 1024, chunkBytes);

    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(12 * 1024, chunkBytes);
    EXPECT_EQ(0, chunkBytes % (6 * 1024));
    EXPECT_EQ(0, chunkBytes % (4 * 1024));
}

TEST(MemfdDataBackendTest, RejectsGeometryWithoutAlignedChunk)
{
    size_t chunkBytes = 0;
    auto status = MemfdDataBackendTestAccess::ResolveRegistrationChunkBytes(
        /*rankStride=*/48 * 1024, /*slotSize=*/12 * 1024, /*pageSize=*/8 * 1024,
        /*maxBytes=*/16 * 1024, chunkBytes);

    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(0, chunkBytes);
}

TEST(MemfdDataBackendTest, ResolvesDeviceAliasWithinEachRegistration)
{
    MemfdDataBackend backend("alias_test");
    auto* firstAlias = reinterpret_cast<void*>(static_cast<uintptr_t>(0x100000));
    auto* secondAlias = reinterpret_cast<void*>(static_cast<uintptr_t>(0x300000));
    MemfdDataBackendTestAccess::SeedRegistrations(backend, /*rankStride=*/32 * 1024,
                                                  /*chunkBytes=*/16 * 1024, firstAlias,
                                                  secondAlias);

    EXPECT_EQ(reinterpret_cast<void*>(static_cast<uintptr_t>(0x101000)),
              backend.HostMappedDeviceAddrOf(0, 4 * 1024));
    EXPECT_EQ(reinterpret_cast<void*>(static_cast<uintptr_t>(0x301000)),
              backend.HostMappedDeviceAddrOf(0, 20 * 1024));
    EXPECT_EQ(nullptr, backend.HostMappedDeviceAddrOf(0, 32 * 1024));
}

}  // namespace

}  // namespace UC::Cache2
