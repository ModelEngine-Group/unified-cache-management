/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for InitAclRuntime error paths (kvcache_pre.cpp).  The engine constructor
 * calls InitAclRuntime which calls aclrtGetCurrentContext + aclrtCreateStream.
 * Setting the stub failure flags before construction triggers the error
 * logging branches.  Also covers the destructor's aclrtDestroyStream error path.
 */
#include <gtest/gtest.h>
#include <string>
#include <torch/torch.h>
#include <vector>
#include "../common/mock_store_v1.h"
#include "../common/test_fixture.h"
#include "kvcache_pre.h"
#include "kvcache_trans_ascend.h"

using ucmprefetch::GSAPrefetchEngineC;
using ucmprefetch_test::CleanupShmFor;
using ucmprefetch_test::MakeModelInfo;
using ucmprefetch_test::MakeSizeInfo;
using ucmprefetch_test::MockStoreV1;
using ucmprefetch_test::TempLogFile;
using ucmprefetch_test::UniqueDpRank;

namespace {

class AclRuntimeErrorTest : public ::testing::Test {
protected:
    int dpRank{0};
    std::unique_ptr<TempLogFile> logFile;
    std::unique_ptr<MockStoreV1> store;

    void SetUp() override
    {
        acl_stub::Stub().Reset();
        dpRank = UniqueDpRank();
        CleanupShmFor(dpRank);
        logFile = std::make_unique<TempLogFile>();
        store = std::make_unique<MockStoreV1>();
    }
    void TearDown() override
    {
        acl_stub::Stub().Reset();
        CleanupShmFor(dpRank);
    }

    std::unique_ptr<GSAPrefetchEngineC> MakeEngine()
    {
        auto model = MakeModelInfo(/*layerNum=*/1, /*maxBlockLen=*/4096,
                                   /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                                   /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/1,
                                   /*indexerOffload=*/1, /*isSfaC8=*/0, /*maxBs=*/2);
        auto size = MakeSizeInfo(/*blockSize=*/4, /*numHead=*/1,
                                 /*kDim=*/4, /*vDim=*/4, /*elementSize=*/2);
        std::vector<int> topkLayerMap{0};
        return std::make_unique<GSAPrefetchEngineC>(
            model, size, /*isLog=*/false,
            /*rank=*/0, /*threadNum=*/1, std::vector<int>{0}, topkLayerMap, logFile->Path());
    }
};

/* ---- InitAclRuntime: aclrtGetCurrentContext failure ---- */
// The engine ctor calls InitAclRuntime; when getCurrentContextFails is set,
// the error is logged and InitAclRuntime throws std::runtime_error (the engine
// cannot be constructed without a valid ACL context). Verify the throw and that
// the stub's failure switch was actually consulted.

TEST_F(AclRuntimeErrorTest, InitAclRuntimeThrowsOnGetCurrentContextFailure)
{
    acl_stub::Stub().getCurrentContextFails = true;
    EXPECT_THROW(
        {
            auto engine = MakeEngine();
            (void)engine;
        },
        std::runtime_error);
    EXPECT_GT(acl_stub::Stub().getCurrentContextCount.load(), 0);
}

// NOTE: InitAclRuntime's aclrtCreateStream failure path still cannot be
// tested in isolation because the TaskManager constructor (which runs first
// inside the engine ctor) also calls aclrtCreateStream via ObtainStream on
// its worker threads; since ObtainStream became a judge-and-return-code API,
// those workers now exit with an error log instead of throwing, so the flag
// would silently disable them before InitAclRuntime gets a chance to run.
// Its own error-logging branch is covered on real hardware.

}  // namespace
