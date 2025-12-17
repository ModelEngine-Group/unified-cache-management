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
#include <fmt/format.h>
#include <fmt/ranges.h>
#include "detail/path_base.h"
#include "detail/types_helper.h"
#include "posix/cc/posix_file.h"
#include "posix/cc/space_manager.h"

class UCPosixSpaceManagerTest : public UC::Test::Detail::PathBase {};

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
    config.storageBackends.push_back(this->Path());
    auto s = spaceMgr.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    auto activated = spaceMgr.GetLayout()->DataFilePath(blockId, true);
    ASSERT_EQ(activated, fmt::format("{}temp/{:02x}", this->Path(), fmt::join(blockId, "")));
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    ASSERT_EQ(PosixFile{activated}.Open(PosixFile::OpenFlag::CREATE), UC::Status::OK());
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::OK());
    ASSERT_EQ(spaceMgr.Lookup(&blockId, 1), std::vector<uint8_t>{false});
    ASSERT_EQ(spaceMgr.GetLayout()->CommitFile(blockId, true), UC::Status::OK());
    ASSERT_EQ(spaceMgr.Lookup(&blockId, 1), std::vector<uint8_t>{true});
    ASSERT_EQ(PosixFile{activated}.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    auto archived = spaceMgr.GetLayout()->DataFilePath(blockId, false);
    ASSERT_EQ(archived, fmt::format("{}data/{:02x}", this->Path(), fmt::join(blockId, "")));
    ASSERT_EQ(PosixFile{archived}.Access(PosixFile::AccessMode::EXIST), UC::Status::OK());
}
