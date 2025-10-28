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
#include "file/file.h"
#include "space/space_recycle.h"
#include "space/space_manager.h"
#include "thread/latch.h"

namespace UC {

void DoRecycle(const SpaceLayout* layout, const uint32_t recycleNum,
               SpaceRecycle::RecycleOneBlockDone done);
}

class UCSpaceRecycleTest : public UC::PathBase {
protected:
    using OpenFlag = UC::IFile::OpenFlag;
    using AccessMode = UC::IFile::AccessMode;

    void NewBlock(const UC::SpaceLayout* layout, const std::string& id)
    {
        std::string parent = layout->DataFileParent(id, false);
        UC::File::MkDir(parent);
        std::string path = layout->DataFilePath(id, false);
        auto f = UC::File::Make(path);
        f->Open(OpenFlag::CREATE | OpenFlag::READ_WRITE);
    }

    bool ExistBlock(const UC::SpaceLayout* layout, const std::string& id)
    {
        std::string path = layout->DataFilePath(id, false);
        return UC::File::Access(path, AccessMode::EXIST).Success();
    }

    void UpdateBlock(const UC::SpaceLayout* layout, const std::string& id)
    {
        struct utimbuf newTime;
        auto tp = time(nullptr) + 3600;
        newTime.modtime = tp;
        newTime.actime = tp;
        std::string path = layout->DataFilePath(id, false);
        utime(path.c_str(), &newTime);
    }
};

TEST_F(UCSpaceRecycleTest, TriggerRecycle)
{
    size_t blocksize = 1024 * 1024;
    UC::SpaceManager spaceMgr;
    ASSERT_EQ(spaceMgr.Setup({this->Path()}, blocksize, false, blocksize * 5), UC::Status::OK());
    const UC::SpaceLayout* layout = spaceMgr.GetSpaceLayout();
    std::string block1 = "a1b2c3d4e5f6789012345678901234ab";
    NewBlock(layout, block1);
    ASSERT_TRUE(ExistBlock(layout, block1));

    std::string block2 = "a2b2c3d4e5f6789012345678901234ab";
    NewBlock(layout, block2);
    ASSERT_TRUE(ExistBlock(layout, block2));

    UpdateBlock(layout, block1);
    UC::SpaceRecycle recycle;
    UC::Latch waiter{1};

    ASSERT_TRUE(recycle.Setup(layout, 10, [&waiter] { waiter.Done([]{}); }).Success());
    recycle.Trigger();
    waiter.Wait();
    EXPECT_TRUE(ExistBlock(layout, block1));
    EXPECT_FALSE(ExistBlock(layout, block2));
}

TEST_F(UCSpaceRecycleTest, DoRecycle)
{
    size_t blocksize = 1024 * 1024;
    UC::SpaceManager spaceMgr;
    ASSERT_EQ(spaceMgr.Setup({this->Path()}, blocksize, false, blocksize * 5), UC::Status::OK());
    const UC::SpaceLayout* layout = spaceMgr.GetSpaceLayout();
    std::string recycleBlocks[] = {
        "a1b2c3d4e5f6789012345678901234ab",
        "a2b2c3d4e5f6789012345678901234ab",
        "a3b2c3d4e5f6789012345678901234ab"
    };
    std::string remainBlocks[] = {
        "b1b2c3d4e5f6789012345678901234ab",
        "b2b2c3d4e5f6789012345678901234ab",
        "b3b2c3d4e5f6789012345678901234ab"
    };
    for (auto &id: remainBlocks)
    {
        NewBlock(layout, id);
        ASSERT_TRUE(ExistBlock(layout, id));
    }
    for (auto &id: recycleBlocks)
    {
        NewBlock(layout, id);
        ASSERT_TRUE(ExistBlock(layout, id));
    }
    for (auto &id: remainBlocks) { UpdateBlock(layout, id); }
    UC::DoRecycle(layout, 3, nullptr);
    for (auto &id: remainBlocks) { EXPECT_TRUE(ExistBlock(layout, id)); }
    for (auto &id: recycleBlocks) { EXPECT_FALSE(ExistBlock(layout, id)); }
}