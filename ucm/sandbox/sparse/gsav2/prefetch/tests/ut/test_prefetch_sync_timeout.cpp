/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for L-9: 非 rank0 等待 rank0 发布 prefetch 结果的超时机制。
 *   - ResolvePrefetchSyncTimeoutUs：环境变量 UCM_PREFETCH_SYNC_TIMEOUT_MS
 *     的解析（默认 30s / 合法覆盖 / 非法回落）。
 *   - SyncPrefetchResultsFromRank0：版本预发布 → 立即返回；版本永不发布 +
 *     1ms 超时覆盖 → ERROR 日志 + KVPackAbort（EXPECT_DEATH）。
 * rank=1 引擎依赖 rank=0 引擎先创建 KVPack 控制面与结果 shm（构造与
 * SetKVPack 均为 follower 语义），故每个用例先建 rank0 再建 rank1。
 */
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>
#include "../common/mock_store_v1.h"
#include "../common/test_fixture.h"
#include "kvcache_pre.h"
#include "kvcache_pre_detail.h"
#include "kvcache_task_manger.h"
#include "kvcache_trans_ascend.h"

using ::TaskInfo;
using ucmprefetch::GSAPrefetchEngineC;
using ucmprefetch::PrefetchShmLayout;
using ucmprefetch::kPrefetchSyncDefaultTimeoutMs;
using ucmprefetch_test::CleanupShmFor;
using ucmprefetch_test::MakeModelInfo;
using ucmprefetch_test::MakeSizeInfo;
using ucmprefetch_test::MockStoreV1;
using ucmprefetch_test::TempLogFile;
using ucmprefetch_test::UniqueDpRank;

namespace {

/* 环境变量用后即清，避免污染同进程内的其他用例。 */
class EnvTimeoutGuard {
public:
    EnvTimeoutGuard() { unsetenv("UCM_PREFETCH_SYNC_TIMEOUT_MS"); }
    ~EnvTimeoutGuard() { unsetenv("UCM_PREFETCH_SYNC_TIMEOUT_MS"); }
};

/* ---- ResolvePrefetchSyncTimeoutUs 环境变量解析 ---- */

TEST(PrefetchSyncTimeoutEnv, DefaultIs30SecondsWhenEnvUnset)
{
    EnvTimeoutGuard guard;
    EXPECT_EQ(ucmprefetch::ResolvePrefetchSyncTimeoutUs(),
              kPrefetchSyncDefaultTimeoutMs * 1000ULL);
    EXPECT_EQ(kPrefetchSyncDefaultTimeoutMs, 30000ULL);
}

TEST(PrefetchSyncTimeoutEnv, OverrideInMilliseconds)
{
    EnvTimeoutGuard guard;
    setenv("UCM_PREFETCH_SYNC_TIMEOUT_MS", "50", 1);
    EXPECT_EQ(ucmprefetch::ResolvePrefetchSyncTimeoutUs(), 50000ULL);
}

TEST(PrefetchSyncTimeoutEnv, NonNumericFallsBackToDefault)
{
    EnvTimeoutGuard guard;
    setenv("UCM_PREFETCH_SYNC_TIMEOUT_MS", "not-a-number", 1);
    EXPECT_EQ(ucmprefetch::ResolvePrefetchSyncTimeoutUs(),
              kPrefetchSyncDefaultTimeoutMs * 1000ULL);
}

TEST(PrefetchSyncTimeoutEnv, NonPositiveFallsBackToDefault)
{
    EnvTimeoutGuard guard;
    setenv("UCM_PREFETCH_SYNC_TIMEOUT_MS", "0", 1);
    EXPECT_EQ(ucmprefetch::ResolvePrefetchSyncTimeoutUs(),
              kPrefetchSyncDefaultTimeoutMs * 1000ULL);
    setenv("UCM_PREFETCH_SYNC_TIMEOUT_MS", "-5", 1);
    EXPECT_EQ(ucmprefetch::ResolvePrefetchSyncTimeoutUs(),
              kPrefetchSyncDefaultTimeoutMs * 1000ULL);
}

/* ---- SyncPrefetchResultsFromRank0 等待环 ---- */

class SyncPrefetchResultsTest : public ::testing::Test {
protected:
    int dpRank{0};
    std::unique_ptr<TempLogFile> logFile;
    std::unique_ptr<MockStoreV1> store;
    std::unique_ptr<GSAPrefetchEngineC> rank0Engine;
    std::unique_ptr<GSAPrefetchEngineC> rank1Engine;

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
        rank1Engine.reset();
        rank0Engine.reset();
        store.reset();
        acl_stub::Stub().Reset();
        CleanupShmFor(dpRank);
    }

    /* rank0 先建（创建 KVPack 控制面 + 结果 shm），rank1 跟随（attach）。 */
    void SetupEngines()
    {
        auto model = MakeModelInfo(/*layerNum=*/1, /*maxBlockLen=*/4096,
                                   /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                                   /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/2,
                                   /*indexerOffload=*/1, /*isSfaC8=*/0, /*maxBs=*/2);
        auto size = MakeSizeInfo(/*blockSize=*/4, /*numHead=*/1,
                                 /*kDim=*/4, /*vDim=*/4, /*elementSize=*/2);
        std::vector<int> topkLayerMap{0};
        rank0Engine = std::make_unique<GSAPrefetchEngineC>(
            model, size, /*isLog=*/false,
            /*rank=*/0, /*threadNum=*/1, std::vector<int>{0}, topkLayerMap,
            logFile->Path());
        rank1Engine = std::make_unique<GSAPrefetchEngineC>(
            model, size, /*isLog=*/false,
            /*rank=*/1, /*threadNum=*/1, std::vector<int>{0}, topkLayerMap,
            logFile->Path());
        rank0Engine->SetStoreV1(store.get());
        rank1Engine->SetStoreV1(store.get());
        auto kHost = torch::zeros({1, 64, 4}, torch::kFloat32);
        auto vHost = torch::zeros({1, 64, 4}, torch::kFloat32);
        auto kDevice = torch::zeros({1, 64, 4}, torch::kFloat32);
        auto vDevice = torch::zeros({1, 64, 4}, torch::kFloat32);
        auto loadIdxHost = torch::zeros({1, 64}, torch::kInt64);
        auto loadIdxDevice = torch::zeros({1, 64}, torch::kInt64);
        rank0Engine->SetKvBuffer(kHost, vHost, kDevice, vDevice, loadIdxHost, loadIdxDevice);
        rank1Engine->SetKvBuffer(kHost, vHost, kDevice, vDevice, loadIdxHost, loadIdxDevice);
        ASSERT_NE(rank1Engine->mKVPackControl, nullptr);
    }
};

TEST_F(SyncPrefetchResultsTest, PrePublishedVersionReturnsImmediately)
{
    SetupEngines();
    uint32_t versionBuf = 0;
    /* 版本预发布：等待环零等待通过，无超时误杀。 */
    const uint32_t expected = rank1Engine->mPrefetchResultVersion.load() + 1;
    __atomic_store_n(&versionBuf, expected, __ATOMIC_RELEASE);
    TaskInfo taskInfo{};  // 空 req 列表：ReadPrefetchResults 为 no-op
    PrefetchShmLayout layout{};
    EXPECT_EQ(rank1Engine->SyncPrefetchResultsFromRank0(taskInfo, layout, &versionBuf), 0);
    EXPECT_EQ(rank1Engine->mPrefetchResultVersion.load(), expected);
}

TEST_F(SyncPrefetchResultsTest, VersionNeverPublishedTimesOutAndAborts)
{
    SetupEngines();
    EnvTimeoutGuard guard;
    /* 1ms 超时覆盖：等待环快速耗尽 → ERROR 日志 + KVPackAbort(SIGABRT)。 */
    setenv("UCM_PREFETCH_SYNC_TIMEOUT_MS", "1", 1);
    uint32_t versionBuf = 0;
    const uint32_t neverExpected = rank1Engine->mPrefetchResultVersion.load() + 2;
    __atomic_store_n(&versionBuf, neverExpected, __ATOMIC_RELEASE);
    TaskInfo taskInfo{};
    PrefetchShmLayout layout{};
    EXPECT_DEATH(rank1Engine->SyncPrefetchResultsFromRank0(taskInfo, layout, &versionBuf),
                 "");
}

}  // namespace
