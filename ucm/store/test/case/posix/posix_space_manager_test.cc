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
#include <chrono>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <sys/stat.h>
#include <thread>
#include <utime.h>
#include "detail/path_base.h"
#include "detail/types_helper.h"
#include "posix/cc/posix_file.h"
#include "posix/cc/space_manager.h"
#include "type/random_block_id.h"

class UCPosixSpaceManagerTest : public UC::Test::Detail::PathBase {};

TEST(RandomBlockIdTest, GeneratesDistinctIds)
{
    EXPECT_NE(UC::Detail::RandomBlockId(), UC::Detail::RandomBlockId());
}

TEST_F(UCPosixSpaceManagerTest, SetStorageBackends)
{
    using namespace UC::PosixStore;
    {
        SpaceManager spaceMgr;
        auto invalidPath = this->Path() + "invalid";
        Config config;
        config.storageBackends.push_back(std::move(invalidPath));
        auto s = spaceMgr.Setup(config);
        ASSERT_EQ(s, UC::Status::OsApiError());
    }
    {
        SpaceManager spaceMgr;
        auto validPath = this->Path();
        auto invalidPath = this->Path() + "invalid";
        Config config;
        config.storageBackends.push_back(std::move(validPath));
        config.storageBackends.push_back(std::move(invalidPath));
        auto s = spaceMgr.Setup(config);
        ASSERT_EQ(s, UC::Status::NotFound());
    }
    {
        SpaceManager spaceMgr;
        Config config;
        config.storageBackends.push_back(this->Path());
        config.storageBackends.push_back(this->Path());
        auto s = spaceMgr.Setup(config);
        ASSERT_EQ(s, UC::Status::OK());
    }
}

TEST_F(UCPosixSpaceManagerTest, DataFilePath)
{
    using namespace UC::PosixStore;
    SpaceManager spaceMgr;
    Config config;
    config.dataDirShardBytes = 0;
    config.storageBackends.push_back(this->Path());
    auto s = spaceMgr.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    auto activated = spaceMgr.GetLayout()->ActivatedFilePath(Path(), blockId);
    ASSERT_EQ(activated, fmt::format("{}data/{:02x}.tmp", this->Path(), fmt::join(blockId, "")));
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    ASSERT_EQ(PosixFile{activated}.Open(PosixFile::OpenFlag::CREATE), UC::Status::OK());
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::OK());
    ASSERT_EQ(spaceMgr.Lookup(&blockId, 1).Value(), std::vector<uint8_t>{false});
    ASSERT_EQ(spaceMgr.GetLayout()->CommitFile(Path(), blockId, true), UC::Status::OK());
    ASSERT_EQ(spaceMgr.Lookup(&blockId, 1).Value(), std::vector<uint8_t>{true});
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    auto archived = spaceMgr.GetLayout()->ArchivedFilePath(Path(), blockId);
    ASSERT_EQ(archived, fmt::format("{}data/{:02x}", this->Path(), fmt::join(blockId, "")));
    ASSERT_EQ(PosixFile{archived}.Access(PosixFile::AccessMode::EXIST), UC::Status::OK());
}

TEST_F(UCPosixSpaceManagerTest, ShardFilePath)
{
    using namespace UC::PosixStore;
    SpaceManager spaceMgr;
    Config config;
    config.dataDirShardBytes = 2;
    config.storageBackends.push_back(this->Path());
    auto s = spaceMgr.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    const auto& file = fmt::format("{:02x}", fmt::join(blockId, ""));
    const auto& shard = file.substr(0, config.dataDirShardBytes);
    auto activated = spaceMgr.GetLayout()->ActivatedFilePath(Path(), blockId);
    ASSERT_EQ(activated, fmt::format("{}{}/{}.tmp", this->Path(), shard, file));
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    ASSERT_EQ(PosixFile{activated}.Open(PosixFile::OpenFlag::CREATE), UC::Status::OK());
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::OK());
    ASSERT_EQ(spaceMgr.Lookup(&blockId, 1).Value(), std::vector<uint8_t>{false});
    ASSERT_EQ(spaceMgr.GetLayout()->CommitFile(Path(), blockId, true), UC::Status::OK());
    ASSERT_EQ(spaceMgr.Lookup(&blockId, 1).Value(), std::vector<uint8_t>{true});
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    auto archived = spaceMgr.GetLayout()->ArchivedFilePath(Path(), blockId);
    ASSERT_EQ(archived, fmt::format("{}{}/{}", this->Path(), shard, file));
    ASSERT_EQ(PosixFile{archived}.Access(PosixFile::AccessMode::EXIST), UC::Status::OK());
}

TEST_F(UCPosixSpaceManagerTest, Lookup)
{
    using namespace UC::PosixStore;
    SpaceManager spaceMgr;
    Config config;
    config.dataDirShardBytes = 0;
    config.lookupConcurrency = 2;
    config.storageBackends.push_back(Path());
    ASSERT_TRUE(spaceMgr.Setup(config).Success());
    std::vector<UC::Detail::BlockId> blocks(3);
    std::for_each(blocks.begin(), blocks.end(), [](auto& block) {
        block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    });
    {
        auto foundIdx = spaceMgr.LookupOnPrefix(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(foundIdx, -1);
        auto ReverseIdx = spaceMgr.LookupOnReverse(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(ReverseIdx, -1);
        auto founds = spaceMgr.Lookup(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(founds.size(), blocks.size());
        std::for_each(founds.begin(), founds.end(), [](auto found) { ASSERT_FALSE(found); });
    }
    std::for_each(blocks.begin(), blocks.end(), [&](const auto& block) {
        auto archived = spaceMgr.GetLayout()->ArchivedFilePath(Path(), block);
        ASSERT_EQ(PosixFile{archived}.Open(PosixFile::OpenFlag::CREATE), UC::Status::OK());
    });
    {
        auto foundIdx = spaceMgr.LookupOnPrefix(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(foundIdx, 2);
        auto ReverseIdx = spaceMgr.LookupOnReverse(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(ReverseIdx, 2);
        auto founds = spaceMgr.Lookup(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(founds.size(), blocks.size());
        std::for_each(founds.begin(), founds.end(), [](auto found) { ASSERT_TRUE(found); });
    }
    auto pos = blocks.begin();
    std::advance(pos, 2);
    blocks.insert(pos, UC::Test::Detail::TypesHelper::MakeBlockIdRandomly());
    {
        auto foundIdx = spaceMgr.LookupOnPrefix(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(foundIdx, 1);
        auto ReverseIdx = spaceMgr.LookupOnReverse(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(ReverseIdx, 3);
        auto founds = spaceMgr.Lookup(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(founds.size(), blocks.size());
        std::vector<uint8_t> expected{true, true, false, true};
        ASSERT_TRUE(founds == expected);
    }
}

TEST_F(UCPosixSpaceManagerTest, PrimaryBackendFailsWhenUnhealthy)
{
    using namespace UC::PosixStore;
    Config config;
    config.dataDirShardBytes = 0;
    config.storageBackends = {Path(), Path() + "./"};
    SpaceLayout layout;
    ASSERT_TRUE(layout.Setup(config).Success());
    BackendManager backendMgr;
    ASSERT_TRUE(backendMgr.Setup(config, &layout).Success());
    const auto& backends = backendMgr.Backends();

    // Pick a block whose primary is the second backend, then trip that backend.
    auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    while (backendMgr.GetBackendForBlock(block).Value() != backends[1]) {
        block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    }
    auto attempt = backendMgr.PrimaryBackend(block);
    ASSERT_TRUE(attempt.HasValue());
    ASSERT_EQ(attempt.Value().index, 1);
    for (size_t i = 0; i < config.backendHealth.passiveFailureThreshold; ++i) {
        backendMgr.RecordResult(attempt.Value(), UC::Status::Error());
    }
    // Load can move to the other backend, but dump does not.
    EXPECT_EQ(backendMgr.GetBackendForBlock(block).Value(), backends[0]);
    attempt = backendMgr.PrimaryBackend(block);
    ASSERT_FALSE(attempt.HasValue());
    EXPECT_EQ(attempt.Error(), UC::Status::StoreUnhealthy());
}

TEST_F(UCPosixSpaceManagerTest, PrefetchTouchesFileThroughBatchPool)
{
    using namespace UC::PosixStore;
    SpaceManager spaceMgr;
    Config config;
    config.dataDirShardBytes = 0;
    config.storageBackends.push_back(Path());
    ASSERT_TRUE(spaceMgr.Setup(config).Success());
    auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto archived = spaceMgr.GetLayout()->ArchivedFilePath(Path(), block);
    ASSERT_EQ(PosixFile{archived}.Open(PosixFile::OpenFlag::CREATE), UC::Status::OK());
    constexpr time_t kOldTime = 1000;
    struct utimbuf old{kOldTime, kOldTime};
    ASSERT_EQ(utime(archived.c_str(), &old), 0);

    spaceMgr.Prefetch(&block, 1);
    time_t mtime = kOldTime;
    for (size_t i = 0; i < 200 && mtime == kOldTime; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        struct stat st{};
        ASSERT_EQ(stat(archived.c_str(), &st), 0);
        mtime = st.st_mtime;
    }
    EXPECT_GT(mtime, kOldTime);
}

TEST_F(UCPosixSpaceManagerTest, LookupParallelPreservesOrder)
{
    using namespace UC::PosixStore;
    SpaceManager spaceMgr;
    Config config;
    config.dataDirShardBytes = 0;
    config.lookupConcurrency = 3;
    config.storageBackends.push_back(Path());
    ASSERT_TRUE(spaceMgr.Setup(config).Success());

    constexpr size_t numBlocks = 8;
    std::vector<UC::Detail::BlockId> blocks(numBlocks);
    std::for_each(blocks.begin(), blocks.end(), [](auto& block) {
        block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    });

    const std::vector<uint8_t> expected{true, false, true, false, false, true, false, true};
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (!expected[i]) { continue; }
        auto archived = spaceMgr.GetLayout()->ArchivedFilePath(Path(), blocks[i]);
        ASSERT_EQ(PosixFile{archived}.Open(PosixFile::OpenFlag::CREATE), UC::Status::OK());
    }

    auto founds = spaceMgr.Lookup(blocks.data(), blocks.size());
    ASSERT_TRUE(founds.HasValue());
    ASSERT_EQ(founds.Value(), expected);
}
