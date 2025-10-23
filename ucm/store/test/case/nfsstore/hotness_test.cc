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
#include <filesystem>
#include "hotness/hotness_set.h"
#include "hotness/hotness_timer.h"
#include "cmn/path_base.h"
#include "file/file.h"
#include "space/space_manager.h"

class UCHotnessTest : public UC::PathBase {};

TEST_F(UCHotnessTest, UpdateHotness)
{
    UC::SpaceManager mgr;
    ASSERT_EQ(mgr.Setup({this->Path()}, 1024 * 1024, false), UC::Status::OK());

    std::string block1 = "a1b2c3d4e5f6789012345678901234ab";
    ASSERT_EQ(mgr.NewBlock(block1), UC::Status::OK());
    ASSERT_EQ(mgr.CommitBlock(block1), UC::Status::OK());

    UC::HotnessSet hotness_set;
    hotness_set.Insert(block1);
    auto space_layout = mgr.GetSpaceLayout();
    auto path = space_layout->DataFilePath(block1, false);
    auto currentTime = std::filesystem::last_write_time(path);
    std::filesystem::last_write_time(path, currentTime - std::chrono::seconds(2));
    auto lastTime = std::filesystem::last_write_time(path);
    hotness_set.UpdateHotness(space_layout);
    auto newTime = std::filesystem::last_write_time(path);
    ASSERT_GT(newTime, lastTime);
}