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
#include <utime.h>
#include "cmn/path_base.h"
#include "space/space_manager.h"
#include "file/file.h"

class UCSpaceManagerTest : public UC::PathBase {};

TEST_F(UCSpaceManagerTest, NewBlockTwice)
{
    UC::SpaceManager spaceMgr;
    ASSERT_EQ(spaceMgr.Setup({this->Path()}, 1024 * 1024, false), UC::Status::OK());
    const std::string block1 = "block1";
    ASSERT_FALSE(spaceMgr.LookupBlock(block1));
    ASSERT_EQ(spaceMgr.NewBlock(block1), UC::Status::OK());
    ASSERT_FALSE(spaceMgr.LookupBlock(block1));
    ASSERT_EQ(spaceMgr.NewBlock(block1), UC::Status::DuplicateKey());
    ASSERT_EQ(spaceMgr.CommitBlock(block1), UC::Status::OK());
    ASSERT_TRUE(spaceMgr.LookupBlock(block1));
    ASSERT_EQ(spaceMgr.NewBlock(block1), UC::Status::DuplicateKey());
}

TEST_F(UCSpaceManagerTest, NewBlockTwiceWithTempDir)
{
    UC::SpaceManager spaceMgr;
    ASSERT_EQ(spaceMgr.Setup({this->Path()}, 1024 * 1024, true), UC::Status::OK());
    const std::string block1 = "block1";
    ASSERT_FALSE(spaceMgr.LookupBlock(block1));
    ASSERT_EQ(spaceMgr.NewBlock(block1), UC::Status::OK());
    ASSERT_FALSE(spaceMgr.LookupBlock(block1));
    ASSERT_EQ(spaceMgr.NewBlock(block1), UC::Status::DuplicateKey());
    ASSERT_EQ(spaceMgr.CommitBlock(block1), UC::Status::OK());
    ASSERT_TRUE(spaceMgr.LookupBlock(block1));
    ASSERT_EQ(spaceMgr.NewBlock(block1), UC::Status::DuplicateKey());
}

TEST_F(UCSpaceManagerTest, CreateBlockWhenNoSpace)
{
    UC::SpaceManager spaceMgr;
    size_t blockSize = 1024 * 1024;
    size_t capacity = blockSize;
    ASSERT_EQ(spaceMgr.Setup({this->Path()}, blockSize, false, capacity), UC::Status::OK());
    ASSERT_EQ(spaceMgr.NewBlock("block3"), UC::Status::OK());
    ASSERT_EQ(spaceMgr.NewBlock("block4"), UC::Status::NoSpace());
}

TEST_F(UCSpaceManagerTest, IterAllBlockFile)
{
    constexpr size_t blockSize = 1024 * 1024;
    constexpr size_t capacity = blockSize * 1024;
    UC::SpaceManager spaceMgr;
    ASSERT_EQ(spaceMgr.Setup({this->Path()}, blockSize, false, capacity), UC::Status::OK());
    const std::string block1 = "a1b2c3d4e5f6789012345678901234ab";
    const std::string block2 = "a2b2c3d4e5f6789012345678901234ab";
    const std::string block3 = "a3b2c3d4e5f6789012345678901234ab";
    ASSERT_EQ(spaceMgr.NewBlock(block1),  UC::Status::OK());
    ASSERT_EQ(spaceMgr.NewBlock(block2),  UC::Status::OK());
    ASSERT_EQ(spaceMgr.NewBlock(block3),  UC::Status::OK());
    auto layout = spaceMgr.GetSpaceLayout();
    auto iter = layout->CreateFilePathIterator();
    size_t count = 0;
    while (!layout->NextDataFilePath(iter).empty()) { count++; }
    ASSERT_EQ(count, 0);
    ASSERT_EQ(spaceMgr.CommitBlock(block1), UC::Status::OK());
    ASSERT_EQ(spaceMgr.CommitBlock(block2), UC::Status::OK());
    ASSERT_EQ(spaceMgr.CommitBlock(block3), UC::Status::OK());
    iter = layout->CreateFilePathIterator();
    count = 0;
    while (!layout->NextDataFilePath(iter).empty()) { count++; }
    ASSERT_EQ(count, 3);
}

TEST_F(UCSpaceManagerTest, NewBlockReuseIfActiveAccessedLongAgo)
{
    UC::SpaceManager spaceMgr;
    constexpr size_t blockSize = 1024 * 1024;
    constexpr size_t capacity = blockSize * 1024;
    ASSERT_EQ(spaceMgr.Setup({this->Path()}, blockSize, false, capacity), UC::Status::OK());
    const auto* layout = spaceMgr.GetSpaceLayout();
    ASSERT_NE(layout, nullptr);

    const std::string block1 = "a1b2c3d4e5f6789012345678901234ab";
    auto parent = UC::File::Make(layout->DataFileParent(block1, /*activated=*/true));
    ASSERT_NE(parent, nullptr);
    ASSERT_EQ(parent->MkDir(), UC::Status::OK());

    const auto activePath = layout->DataFilePath(block1, /*activated=*/true);
    auto activeFile = UC::File::Make(activePath);
    ASSERT_NE(activeFile, nullptr);
    ASSERT_EQ(activeFile->Open(UC::IFile::OpenFlag::CREATE | UC::IFile::OpenFlag::READ_WRITE), UC::Status::OK());
    activeFile->Close();

    // NewBlock should return DuplicateKey because the file is recent
    ASSERT_EQ(spaceMgr.NewBlock(block1), UC::Status::DuplicateKey());

    // Set atime to 10 minutes ago so it is not considered recent
    struct utimbuf newTime;
    auto tp = time(nullptr) - 600;
    newTime.modtime = tp;
    newTime.actime = tp;
    utime(activePath.c_str(), &newTime);
    ASSERT_EQ(spaceMgr.NewBlock(block1), UC::Status::OK());
}