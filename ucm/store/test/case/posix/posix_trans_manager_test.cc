/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include <atomic>
#include <future>
#include "detail/data_generator.h"
#include "detail/path_base.h"
#include "detail/types_helper.h"
#include "posix/cc/backend_manager.h"
#include "posix/cc/trans_manager.h"
#include "space_layout.h"

class UCPosixTransManagerTest : public UC::Test::Detail::PathBase {};

TEST_F(UCPosixTransManagerTest, EngineCallbacksReclaimCompletedTasks)
{
    using namespace UC::PosixStore;
    Config config;
    config.tensorSize = 4096;
    config.shardSize = config.tensorSize;
    config.blockSize = config.shardSize;
    config.storageBackends = {Path()};
    config.dataDirShardBytes = 0;
    config.openConcurrency = 2;
    config.commitConcurrency = 1;
    config.dataTransConcurrency = 2;
    config.timeoutMs = 3000;
    SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), UC::Status::OK());
    UC::PosixStore::BackendManager backendMgr;
    ASSERT_TRUE(backendMgr.Setup(config, &layout).Success());
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.GenerateRandom();
    auto checkCallback = [&](auto& engine) {
        ASSERT_EQ(engine.Setup(config, &layout, backendMgr.Backends().front()), UC::Status::OK());
        const auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        for (size_t round = 0; round < 3; ++round) {
            SCOPED_TRACE(round);
            const auto id =
                round == 2 ? UC::Test::Detail::TypesHelper::MakeBlockIdRandomly() : block;
            UC::Detail::TaskDesc desc;
            desc.push_back({id, 0, {data.Buffer()}});
            TransTask task{round == 0 ? TransTask::Type::DUMP : TransTask::Type::LOAD,
                           std::move(desc)};
            auto done = std::make_shared<std::promise<UC::Status>>();
            auto result = done->get_future();
            auto calls = std::make_shared<std::atomic<size_t>>(0);
            task.onComplete = [done, calls](UC::Status status) {
                if (calls->fetch_add(1) == 0) { done->set_value(status); }
            };
            auto submitted = engine.Submit(std::move(task));
            ASSERT_TRUE(submitted.HasValue());
            ASSERT_EQ(result.wait_for(std::chrono::seconds(2)), std::future_status::ready);
            EXPECT_EQ(result.get(), round == 2 ? UC::Status::NotFound() : UC::Status::OK());
            auto checked = engine.Check(submitted.Value());
            ASSERT_FALSE(checked.HasValue());
            EXPECT_EQ(checked.Error(), UC::Status::NotFound());
            EXPECT_EQ(calls->load(), 1);
        }
    };
    IoEngineAio aio;
    checkCallback(aio);
    IoEnginePsync psync;
    checkCallback(psync);
}

TEST_F(UCPosixTransManagerTest, TransBlock)
{
    using namespace UC::PosixStore;
    Config config;
    config.tensorSize = 32768;
    config.shardSize = config.tensorSize;
    config.blockSize = config.shardSize;
    config.storageBackends.push_back(Path());
    UC::PosixStore::SpaceLayout layout;
    ASSERT_TRUE(layout.Setup(config).Success());
    UC::PosixStore::BackendManager backendMgr;
    ASSERT_TRUE(backendMgr.Setup(config, &layout).Success());
    TransManager transMgr;
    auto s = transMgr.Setup(config, &layout, &backendMgr);
    ASSERT_EQ(s, UC::Status::OK());
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t nBlocks = 1;
    UC::Test::Detail::DataGenerator data1{nBlocks, config.blockSize};
    data1.GenerateRandom();
    UC::Detail::TaskDesc desc1;
    desc1.brief = "Dump";
    desc1.push_back(UC::Detail::Shard{block, 0, {data1.Buffer()}});
    auto handle1 = transMgr.Submit({TransTask::Type::DUMP, desc1});
    ASSERT_TRUE(handle1.HasValue());
    s = transMgr.Wait(handle1.Value());
    ASSERT_EQ(s, UC::Status::OK());
    UC::Test::Detail::DataGenerator data2{nBlocks, config.blockSize};
    data2.Generate();
    UC::Detail::TaskDesc desc2;
    desc2.brief = "Load";
    desc2.push_back(UC::Detail::Shard{block, 0, {data2.Buffer()}});
    auto handle2 = transMgr.Submit({TransTask::Type::LOAD, desc2});
    ASSERT_TRUE(handle2.HasValue());
    s = transMgr.Wait(handle2.Value());
    ASSERT_EQ(s, UC::Status::OK());
    ASSERT_EQ(data1.Compare(data2), 0);
}

TEST_F(UCPosixTransManagerTest, TransBlockLayerWise)
{
    using namespace UC::PosixStore;
    constexpr size_t nShards = 8;
    Config config;
    config.tensorSize = 32768;
    config.shardSize = config.tensorSize;
    config.blockSize = config.shardSize * nShards;
    config.storageBackends.push_back(Path());
    UC::PosixStore::SpaceLayout layout;
    ASSERT_TRUE(layout.Setup(config).Success());
    UC::PosixStore::BackendManager backendMgr;
    ASSERT_TRUE(backendMgr.Setup(config, &layout).Success());
    TransManager transMgr;
    auto s = transMgr.Setup(config, &layout, &backendMgr);
    ASSERT_EQ(s, UC::Status::OK());
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    auto data1 = UC::Test::Detail::TypesHelper::MakeArray<UC::Test::Detail::DataGenerator, nShards>(
        size_t(1), config.tensorSize);
    UC::Detail::TaskDesc desc1;
    desc1.brief = "Dump";
    for (size_t i = 0; i < nShards; i++) {
        auto& d = data1[i];
        d.GenerateRandom();
        desc1.push_back(UC::Detail::Shard{block, i, {d.Buffer()}});
    }
    auto handle1 = transMgr.Submit({TransTask::Type::DUMP, desc1});
    ASSERT_TRUE(handle1.HasValue());
    s = transMgr.Wait(handle1.Value());
    ASSERT_EQ(s, UC::Status::OK());
    auto data2 = UC::Test::Detail::TypesHelper::MakeArray<UC::Test::Detail::DataGenerator, nShards>(
        size_t(1), config.tensorSize);
    UC::Detail::TaskDesc desc2;
    desc2.brief = "Load";
    for (size_t i = 0; i < nShards; i++) {
        auto& d = data2[i];
        d.Generate();
        desc2.push_back(UC::Detail::Shard{block, i, {d.Buffer()}});
    }
    auto handle2 = transMgr.Submit({TransTask::Type::LOAD, desc2});
    ASSERT_TRUE(handle2.HasValue());
    s = transMgr.Wait(handle2.Value());
    ASSERT_EQ(s, UC::Status::OK());
    for (size_t i = 0; i < nShards; i++) { ASSERT_EQ(data1[i].Compare(data2[i]), 0); }
}

TEST_F(UCPosixTransManagerTest, PsyncLoadSubmitAfterCloseFailsCleanly)
{
    using namespace UC::PosixStore;
    Config config;
    config.tensorSize = 32768;
    config.shardSize = config.tensorSize;
    config.blockSize = config.shardSize;
    config.storageBackends.push_back(Path());
    UC::PosixStore::SpaceLayout layout;
    ASSERT_TRUE(layout.Setup(config).Success());
    BackendManager backendMgr;
    ASSERT_TRUE(backendMgr.Setup(config, &layout).Success());
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();

    IoEnginePsync engine;
    ASSERT_EQ(engine.Setup(config, &layout, backendMgr.Backends().front()), UC::Status::OK());
    UC::Detail::TaskDesc dumpDesc;
    dumpDesc.brief = "Dump";
    dumpDesc.push_back(UC::Detail::Shard{block, 0, {data.Buffer()}});
    auto dumpHandle = engine.Submit({TransTask::Type::DUMP, dumpDesc});
    ASSERT_TRUE(dumpHandle.HasValue());
    ASSERT_EQ(engine.Wait(dumpHandle.Value()), UC::Status::OK());

    engine.Close();
    UC::Detail::TaskDesc loadDesc;
    loadDesc.brief = "Load";
    loadDesc.push_back(UC::Detail::Shard{block, 0, {data.Buffer()}});
    auto loadHandle = engine.Submit({TransTask::Type::LOAD, loadDesc});
    ASSERT_TRUE(loadHandle.HasValue());
    EXPECT_TRUE(engine.Wait(loadHandle.Value()).Failure());
}
