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
#include "posix/cc/posix_file.h"
#include "detail/data_generator.h"
#include "detail/path_base.h"

class UCPosixFileTest : public UC::Test::Detail::PathBase {};

TEST_F(UCPosixFileTest, DirCreateAndRemove)
{
    using namespace UC::PosixStore;
    system((std::string("rm -rf ") + this->Path()).c_str());
    PosixFile dir(this->Path());
    ASSERT_EQ(dir.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    ASSERT_EQ(dir.MkDir(), UC::Status::OK());
    ASSERT_EQ(dir.Access(PosixFile::AccessMode::EXIST), UC::Status::OK());
    ASSERT_EQ(dir.Access(PosixFile::AccessMode::READ), UC::Status::OK());
    ASSERT_EQ(dir.Access(PosixFile::AccessMode::WRITE), UC::Status::OK());
    ASSERT_EQ(dir.MkDir(), UC::Status::DuplicateKey());
    ASSERT_EQ(dir.RmDir(), UC::Status::OK());
    ASSERT_EQ(dir.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
}

TEST_F(UCPosixFileTest, FileCreateAndRemove)
{
    using namespace UC::PosixStore;
    PosixFile file(this->Path() + "file");
    ASSERT_EQ(file.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    auto flags = PosixFile::OpenFlag::WRITE_ONLY;
    ASSERT_EQ(file.Open(flags), UC::Status::OsApiError());
    flags |= PosixFile::OpenFlag::CREATE;
    ASSERT_EQ(file.Open(flags), UC::Status::OK());
    flags |= PosixFile::OpenFlag::EXCL;
    ASSERT_EQ(file.Open(flags), UC::Status::DuplicateKey());
    ASSERT_EQ(file.Access(PosixFile::AccessMode::EXIST), UC::Status::OK());
    ASSERT_EQ(file.Access(PosixFile::AccessMode::READ), UC::Status::OK());
    ASSERT_EQ(file.Access(PosixFile::AccessMode::WRITE), UC::Status::OK());
    file.Remove();
    ASSERT_EQ(file.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
}

TEST_F(UCPosixFileTest, FileWriteAndRead)
{
    using namespace UC::PosixStore;
    PosixFile file(this->Path() + "file");
    size_t nPage = 4;
    UC::Test::Detail::DataGenerator data0{nPage};
    UC::Test::Detail::DataGenerator data1{nPage};
    data0.GenerateRandom();
    data1.Generate();
    ASSERT_EQ(file.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    constexpr auto flags = PosixFile::OpenFlag::WRITE_ONLY | PosixFile::OpenFlag::CREATE;
    ASSERT_EQ(file.Open(flags), UC::Status::OK());
    ASSERT_EQ(file.Write(data0.Buffer(), data0.Size(), 0), UC::Status::OK());
    file.Close();
    ASSERT_EQ(file.Open(PosixFile::OpenFlag::READ_ONLY), UC::Status::OK());
    ASSERT_EQ(file.Read(data1.Buffer(), data1.Size(), 0), UC::Status::OK());
    file.Close();
    file.Remove();
    ASSERT_EQ(file.Access(PosixFile::AccessMode::EXIST), UC::Status::NotFound());
    EXPECT_EQ(data0.Compare(data1), 0);
}
