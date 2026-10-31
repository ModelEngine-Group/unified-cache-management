/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the ACL transport wrappers in ascend/kvcache_trans_ascend.cpp.
 * Uses the test stub (kvcache_trans_ascend_stub.h) to drive every success and
 * failure branch without a real Ascend device.
 */
#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>
#include "kvcache_log.h"
#include "kvcache_trans_ascend.h"
#include "kvcache_trans_ascend_stub.h"

namespace {
Logger kLog{};  // disabled

template <typename F>
void ExpectThrow(F&& f)
{
    EXPECT_THROW(f(), std::runtime_error);
}

}  // namespace

class AscendTransTest : public ::testing::Test {
protected:
    void SetUp() override { acl_stub::Stub().Reset(); }
    void TearDown() override { acl_stub::Stub().Reset(); }
};

/* ---- CopyTrans ---- */

TEST_F(AscendTransTest, CopyTransHostToDeviceSucceedsAndCopies)
{
    // CopyTrans takes void** = array of pointers to buffers, NOT a pointer to
    // a single buffer. Each entry is a void* pointing at one buffer of size_bytes.
    int host = 0x1234;
    int dev = 0;
    void* hostPtrs[1] = {reinterpret_cast<void*>(&host)};
    void* devPtrs[1] = {reinterpret_cast<void*>(&dev)};
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x1);
    int32_t ret = CopyTrans(hostPtrs, devPtrs, sizeof(int), 1, "h2d", &stream);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(dev, 0x1234);
    EXPECT_EQ(acl_stub::Stub().memcpyAsyncCount.load(), 1);
}

TEST_F(AscendTransTest, CopyTransDeviceToHostSucceedsAndCopies)
{
    int dev = 0xABCD;
    int host = 0;
    // CopyTrans(host_ptrs_addr, device_ptrs_addr, ...): for d2h the dst is
    // host_ptrs_addr[i], src is device_ptrs_addr[i].
    void* devPtrs[1] = {reinterpret_cast<void*>(&dev)};
    void* hostPtrs[1] = {reinterpret_cast<void*>(&host)};
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x1);
    int32_t ret = CopyTrans(hostPtrs, devPtrs, sizeof(int), 1, "d2h", &stream);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(host, 0xABCD);
}

TEST_F(AscendTransTest, CopyTransFailurePropagatesMinusOne)
{
    acl_stub::Stub().memcpyAsyncFails = true;
    int host = 1, dev = 0;
    void* hostPtrs[1] = {reinterpret_cast<void*>(&host)};
    void* devPtrs[1] = {reinterpret_cast<void*>(&dev)};
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x1);
    int32_t ret = CopyTrans(hostPtrs, devPtrs, sizeof(int), 1, "h2d", &stream);
    EXPECT_EQ(ret, -1);
}

TEST_F(AscendTransTest, CopyTransMultipleEntriesStopsOnFirstError)
{
    int host[2] = {1, 2};
    int dev[2] = {0, 0};
    void* hostPtrs[2] = {reinterpret_cast<void*>(&host[0]), reinterpret_cast<void*>(&host[1])};
    void* devPtrs[2] = {reinterpret_cast<void*>(&dev[0]), reinterpret_cast<void*>(&dev[1])};
    acl_stub::Stub().memcpyAsyncFails = true;
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x1);
    int32_t ret = CopyTrans(hostPtrs, devPtrs, sizeof(int), 2, "h2d", &stream);
    EXPECT_EQ(ret, -1);
    EXPECT_EQ(acl_stub::Stub().memcpyAsyncCount.load(), 1);
}

TEST_F(AscendTransTest, CopyTransZeroNumberSucceeds)
{
    int host = 0, dev = 0;
    void* hostPtrs[1] = {reinterpret_cast<void*>(&host)};
    void* devPtrs[1] = {reinterpret_cast<void*>(&dev)};
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x1);
    int32_t ret = CopyTrans(hostPtrs, devPtrs, sizeof(int), 0, "h2d", &stream);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(acl_stub::Stub().memcpyAsyncCount.load(), 0);
}

TEST_F(AscendTransTest, CopyTransMultipleEntriesCopiesAllInOrder)
{
    // Positive multi-entry case: both entries copy, count == 2, values correct.
    int host[2] = {0xA, 0xB};
    int dev[2] = {0, 0};
    void* hostPtrs[2] = {reinterpret_cast<void*>(&host[0]), reinterpret_cast<void*>(&host[1])};
    void* devPtrs[2] = {reinterpret_cast<void*>(&dev[0]), reinterpret_cast<void*>(&dev[1])};
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x1);
    int32_t ret = CopyTrans(hostPtrs, devPtrs, sizeof(int), 2, "h2d", &stream);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(dev[0], 0xA);
    EXPECT_EQ(dev[1], 0xB);
    EXPECT_EQ(acl_stub::Stub().memcpyAsyncCount.load(), 2);
}

/* ---- ObtainStream / DestroyStream ---- */

TEST_F(AscendTransTest, ObtainStreamCreatesContextAndStreams)
{
    void* stream = nullptr;
    void* context = nullptr;
    EXPECT_NO_THROW(ObtainStream(&stream, &context, /*deviceID=*/0, /*mStreamNum=*/2));
    EXPECT_NE(stream, nullptr);
    EXPECT_NE(context, nullptr);
    EXPECT_EQ(acl_stub::Stub().setDeviceCount.load(), 1);
    EXPECT_EQ(acl_stub::Stub().createContextCount.load(), 1);
    EXPECT_EQ(acl_stub::Stub().createStreamCount.load(), 2);
    DestroyStream(stream, context, 0, 2);
    EXPECT_EQ(acl_stub::Stub().destroyStreamCount.load(), 2);
}

TEST_F(AscendTransTest, ObtainStreamThrowsOnSetDeviceFailure)
{
    acl_stub::Stub().setDeviceFails = true;
    void* stream = nullptr;
    void* context = nullptr;
    ExpectThrow([&] { ObtainStream(&stream, &context, 0, 1); });
}

TEST_F(AscendTransTest, ObtainStreamThrowsOnCreateContextFailure)
{
    acl_stub::Stub().createContextFails = true;
    void* stream = nullptr;
    void* context = nullptr;
    ExpectThrow([&] { ObtainStream(&stream, &context, 0, 1); });
    /* BUG-FOCUS: context memory is leaked on this path. See README. */
}

TEST_F(AscendTransTest, ObtainStreamThrowsOnCreateStreamFailure)
{
    acl_stub::Stub().createStreamFails = true;
    void* stream = nullptr;
    void* context = nullptr;
    ExpectThrow([&] { ObtainStream(&stream, &context, 0, 2); });
    /* BUG-FOCUS: context + partial streams leaked on this path. See README. */
}

/* ---- Synchronize ---- */

TEST_F(AscendTransTest, SynchronizeSucceeds)
{
    aclrtContext ctx = reinterpret_cast<aclrtContext>(0x1);
    aclrtStream streams[2] = {reinterpret_cast<aclrtStream>(0x10),
                              reinterpret_cast<aclrtStream>(0x20)};
    int32_t ret =
        Synchronize(reinterpret_cast<void*>(&ctx), reinterpret_cast<void*>(streams), 2, kLog);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(acl_stub::Stub().setCurrentContextCount.load(), 1);
    EXPECT_EQ(acl_stub::Stub().syncStreamCount.load(), 2);
}

TEST_F(AscendTransTest, SynchronizeSetCurrentContextFailureReturnsError)
{
    acl_stub::Stub().setCurrentContextFails = true;
    aclrtContext ctx = reinterpret_cast<aclrtContext>(0x1);
    aclrtStream streams[1] = {reinterpret_cast<aclrtStream>(0x10)};
    int32_t ret =
        Synchronize(reinterpret_cast<void*>(&ctx), reinterpret_cast<void*>(streams), 1, kLog);
    EXPECT_EQ(ret, -1);
}

TEST_F(AscendTransTest, SynchronizeStreamFailureReturnsError)
{
    acl_stub::Stub().syncStreamFails = true;
    aclrtContext ctx = reinterpret_cast<aclrtContext>(0x1);
    aclrtStream streams[1] = {reinterpret_cast<aclrtStream>(0x10)};
    int32_t ret =
        Synchronize(reinterpret_cast<void*>(&ctx), reinterpret_cast<void*>(streams), 1, kLog);
    EXPECT_EQ(ret, -1);
}

/* ---- IsStreamIdle ---- */

TEST_F(AscendTransTest, IsStreamIdleReturnsTrueForComplete)
{
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x1);
    EXPECT_TRUE(IsStreamIdle(reinterpret_cast<void*>(&stream)));
    EXPECT_EQ(acl_stub::Stub().streamQueryCount.load(), 1);
}

TEST_F(AscendTransTest, IsStreamIdleReturnsFalseOnQueryFailure)
{
    acl_stub::Stub().streamQueryFails = true;
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x1);
    EXPECT_FALSE(IsStreamIdle(reinterpret_cast<void*>(&stream)));
    /* BUG-FOCUS: query failure is silently swallowed. See README. */
}

/* ---- DestroyStream error path ---- */

TEST_F(AscendTransTest, DestroyStreamToleratesDestroyFailure)
{
    acl_stub::Stub().destroyStreamFails = true;
    // DestroyStream free()s the stream/context arrays, so they MUST be
    // heap-allocated (mirroring ObtainStream's malloc). Use a 1-element array.
    aclrtStream* streamArr = static_cast<aclrtStream*>(std::malloc(sizeof(aclrtStream)));
    *streamArr = reinterpret_cast<aclrtStream>(0x1);
    aclrtContext* ctxArr = static_cast<aclrtContext*>(std::malloc(sizeof(aclrtContext)));
    *ctxArr = reinterpret_cast<aclrtContext>(0x2);
    DestroyStream(streamArr, ctxArr, /*deviceID=*/0, /*mStreamNum=*/1);
    /* DestroyStream only logs on failure; ensure no crash. */
    EXPECT_EQ(acl_stub::Stub().destroyStreamCount.load(), 1);
}

/* ---- GetAclDataType / AtTensorToAclTensorShared ---- */

TEST_F(AscendTransTest, GetAclDataTypeMapsKnownScalarTypes)
{
    EXPECT_EQ(GetAclDataType(at::kFloat), ACL_FLOAT);
    EXPECT_EQ(GetAclDataType(at::kBFloat16), ACL_BF16);
    EXPECT_EQ(GetAclDataType(at::kUInt64), ACL_UINT64);
    EXPECT_EQ(GetAclDataType(at::kUInt32), ACL_UINT32);
    EXPECT_EQ(GetAclDataType(at::kHalf), ACL_FLOAT16);
    EXPECT_EQ(GetAclDataType(at::kInt), ACL_INT32);
    EXPECT_EQ(GetAclDataType(at::kLong), ACL_INT64);
    EXPECT_EQ(GetAclDataType(at::kDouble), ACL_DOUBLE);
    EXPECT_EQ(GetAclDataType(at::kBool), ACL_BOOL);
    EXPECT_EQ(GetAclDataType(at::kChar), ACL_INT8);
}

TEST_F(AscendTransTest, GetAclDataTypeUnknownReturnsUndefined)
{
    /* Pick a scalar type outside the switch. at::kQInt8 is int8 quantized. */
    EXPECT_EQ(GetAclDataType(at::kQInt8), ACL_DT_UNDEFINED);
}

TEST_F(AscendTransTest, AtTensorToAclTensorSharedReturnsPointer)
{
    acl_stub::Stub().Reset();
    at::Tensor t = torch::zeros({4}, torch::kFloat);
    aclTensor* acl = AtTensorToAclTensorShared(t);
    EXPECT_NE(acl, nullptr);
    EXPECT_EQ(acl_stub::Stub().createTensorCount.load(), 1);
}

TEST_F(AscendTransTest, AtTensorToAclTensorSharedReturnsNullOnFailure)
{
    acl_stub::Stub().createTensorFails = true;
    at::Tensor t = torch::zeros({4}, torch::kFloat);
    aclTensor* acl = AtTensorToAclTensorShared(t);
    EXPECT_EQ(acl, nullptr);
}

TEST_F(AscendTransTest, ReleaseAclTensorCallsDestroy)
{
    acl_stub::Stub().Reset();
    aclTensor* fake = reinterpret_cast<aclTensor*>(0x1);
    ReleaseAclTensor(fake);
    EXPECT_EQ(acl_stub::Stub().destroyTensorCount.load(), 1);
}
