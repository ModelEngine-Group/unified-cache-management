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
#include "file/file.h"

class UCFileTest : public UC::PathBase {};

TEST_F(UCFileTest, DirCreateAndRemove)
{
    auto path1 = this->Path() + "dir1";
    ASSERT_EQ(UC::File::Access(path1, UC::IFile::AccessMode::EXIST), UC::Status::NotFound());
    ASSERT_EQ(UC::File::MkDir(path1), UC::Status::OK());
    ASSERT_EQ(UC::File::Access(path1, UC::IFile::AccessMode::EXIST), UC::Status::OK());
    auto path2 = this->Path() + "dir2";
    ASSERT_EQ(UC::File::Access(path2, UC::IFile::AccessMode::EXIST), UC::Status::NotFound());
    ASSERT_EQ(UC::File::Rename(path1, path2), UC::Status::OK());
    ASSERT_EQ(UC::File::Access(path1, UC::IFile::AccessMode::EXIST), UC::Status::NotFound());
    ASSERT_EQ(UC::File::Access(path2, UC::IFile::AccessMode::EXIST), UC::Status::OK());
    ASSERT_EQ(UC::File::RmDir(path2), UC::Status::OK());
    ASSERT_EQ(UC::File::Access(path2, UC::IFile::AccessMode::EXIST), UC::Status::NotFound());
}
