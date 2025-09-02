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
#include "cmn/path_base.h"
#include "gc/gc_runner.h"
#include "space/space_manager.h"
#include "status/status.h"

class UCGCTest : public UC::PathBase {};

TEST_F(UCGCTest, BelowThreshold)
{
    UC::SpaceManager spmg;
    auto status = spmg.Setup({this->Path()}, 1024);
    ASSERT_EQ(status, UC::Status::OK());
    status = spmg.NewBlock("1111111111111111");
    ASSERT_EQ(status, UC::Status::OK());
    status = spmg.CommitBlock("1111111111111111");
    ASSERT_EQ(status, UC::Status::OK());
    status = spmg.NewBlock("2222222222222222");
    ASSERT_EQ(status, UC::Status::OK());
    status = spmg.CommitBlock("2222222222222222");
    ASSERT_EQ(status, UC::Status::OK());
    ASSERT_EQ(spmg.GetUsedSpace(), 2048);
    UC::GCRunner gcRunner(4096, 0.25);
    gcRunner();
    ASSERT_EQ(spmg.GetUsedSpace(), 2048);
}

TEST_F(UCGCTest, AboveThreshold)
{
    UC::SpaceManager spmg;
    auto status = spmg.Setup({this->Path()}, 1024);
    ASSERT_EQ(status, UC::Status::OK());
    status = spmg.NewBlock("1111111111111111");
    ASSERT_EQ(status, UC::Status::OK());
    status = spmg.CommitBlock("1111111111111111");
    ASSERT_EQ(status, UC::Status::OK());
    status = spmg.NewBlock("2222222222222222");
    ASSERT_EQ(status, UC::Status::OK());
    status = spmg.CommitBlock("2222222222222222");
    ASSERT_EQ(status, UC::Status::OK());
    status = spmg.NewBlock("3333333333333333");
    ASSERT_EQ(status, UC::Status::OK());
    status = spmg.CommitBlock("3333333333333333");
    ASSERT_EQ(status, UC::Status::OK());
    status = spmg.NewBlock("4444444444444444");
    ASSERT_EQ(status, UC::Status::OK());
    status = spmg.CommitBlock("4444444444444444");
    ASSERT_EQ(status, UC::Status::OK());
    ASSERT_EQ(spmg.GetUsedSpace(), 4096);
    UC::GCRunner gcRunner(4096, 0.25);
    gcRunner();
    ASSERT_EQ(spmg.GetUsedSpace(), 3072);
}
