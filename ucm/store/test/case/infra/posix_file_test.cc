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
#include "file/posix_file.h"

class UCPosixFileTest : public UC::PathBase {
public:
    class Data {
    public:
        explicit Data(const size_t nPage, const size_t pageSize = 4096)
            : nPage_{nPage}, pageSize_{pageSize}, data_{nullptr}
        {
        }
        ~Data()
        {
            if (this->data_) {
                free(this->data_);
                this->data_ = nullptr;
            }
        }
        void Generate()
        {
            this->data_ = malloc(this->Size());
            assert(this->data_ != nullptr);
        }
        void GenerateRandom()
        {
            this->Generate();
            for (size_t i = 0; i < this->nPage_; i++) {
                *(size_t*)((char*)this->data_ + this->pageSize_ * i) = i;
            }
        }
        int32_t Compare(const Data& other)
        {
            if (this->nPage_ < other.nPage_) { return -1; }
            if (this->nPage_ < other.nPage_) { return 1; }
            for (size_t i = 0; i < this->nPage_; i++) {
                auto ret = memcmp((char*)this->data_ + this->pageSize_ * i,
                                  (char*)other.data_ + this->pageSize_ * i, this->pageSize_);
                if (ret != 0) { return ret; }
            }
            return 0;
        }
        size_t Size() const { return this->pageSize_ * this->nPage_; }
        void* Buffer() const { return this->data_; }

    private:
        size_t nPage_;
        size_t pageSize_;
        void* data_;
    };
};

TEST_F(UCPosixFileTest, DirCreateAndRemove)
{
    system((std::string("rm -rf ") + this->Path()).c_str());
    UC::PosixFile dir(this->Path());
    ASSERT_EQ(dir.Access(UC::IFile::AccessMode::EXIST), UC::Status::NotFound());
    ASSERT_EQ(dir.MkDir(), UC::Status::OK());
    ASSERT_EQ(dir.Access(UC::IFile::AccessMode::EXIST), UC::Status::OK());
    ASSERT_EQ(dir.Access(UC::IFile::AccessMode::READ), UC::Status::OK());
    ASSERT_EQ(dir.Access(UC::IFile::AccessMode::WRITE), UC::Status::OK());
    ASSERT_EQ(dir.MkDir(), UC::Status::DuplicateKey());
    ASSERT_EQ(dir.RmDir(), UC::Status::OK());
    ASSERT_EQ(dir.Access(UC::IFile::AccessMode::EXIST), UC::Status::NotFound());
}

TEST_F(UCPosixFileTest, FileCreateAndRemove)
{
    UC::PosixFile file(this->Path() + "file");
    ASSERT_EQ(file.Access(UC::IFile::AccessMode::EXIST), UC::Status::NotFound());
    ASSERT_EQ(file.Open(UC::IFile::OpenFlag::WRITE_ONLY), UC::Status::OsApiError());
    ASSERT_EQ(file.Open(UC::IFile::OpenFlag::WRITE_ONLY | UC::IFile::OpenFlag::CREATE),
              UC::Status::OK());
    ASSERT_EQ(file.Open(UC::IFile::OpenFlag::WRITE_ONLY | UC::IFile::OpenFlag::CREATE |
                        UC::IFile::OpenFlag::EXCL),
              UC::Status::DuplicateKey());
    ASSERT_EQ(file.Access(UC::IFile::AccessMode::EXIST), UC::Status::OK());
    ASSERT_EQ(file.Access(UC::IFile::AccessMode::READ), UC::Status::OK());
    ASSERT_EQ(file.Access(UC::IFile::AccessMode::WRITE), UC::Status::OK());
    file.Remove();
    ASSERT_EQ(file.Access(UC::IFile::AccessMode::EXIST), UC::Status::NotFound());
}

TEST_F(UCPosixFileTest, FileWriteAndRead)
{
    UC::PosixFile file(this->Path() + "file");
    size_t nPage = 4;
    UCPosixFileTest::Data data0{nPage};
    UCPosixFileTest::Data data1{nPage};
    data0.GenerateRandom();
    data1.Generate();
    ASSERT_EQ(file.Access(UC::IFile::AccessMode::EXIST), UC::Status::NotFound());
    ASSERT_EQ(file.Open(UC::IFile::OpenFlag::WRITE_ONLY | UC::IFile::OpenFlag::CREATE),
              UC::Status::OK());
    ASSERT_EQ(file.Write(data0.Buffer(), data0.Size(), 0), UC::Status::OK());
    file.Close();
    ASSERT_EQ(file.Open(UC::IFile::OpenFlag::READ_ONLY), UC::Status::OK());
    ASSERT_EQ(file.Read(data1.Buffer(), data1.Size(), 0), UC::Status::OK());
    file.Close();
    file.Remove();
    ASSERT_EQ(file.Access(UC::IFile::AccessMode::EXIST), UC::Status::NotFound());
    EXPECT_EQ(data0.Compare(data1), 0);
}

TEST_F(UCPosixFileTest, FileMMapAndMUnmap)
{
    UC::Random rd;
    auto fileName = rd.RandomString(20) + ".file";
    UC::PosixFile file1{fileName};
    UC::PosixFile file2{fileName};
    const size_t data = 0xfffffffe;
    const auto openFlags = UC::IFile::OpenFlag::READ_WRITE | UC::IFile::OpenFlag::CREATE;
    void* addr1 = nullptr;
    void* addr2 = nullptr;
    ASSERT_EQ(file1.Path(), file2.Path());
    ASSERT_TRUE(file1.ShmOpen(openFlags).Success());
    ASSERT_TRUE(file1.Truncate(sizeof(data)).Success());
    ASSERT_TRUE(file1.MMap(addr1, sizeof(data), true, true, true).Success());
    file1.Close();
    ASSERT_TRUE(file2.ShmOpen(openFlags).Success());
    ASSERT_TRUE(file2.MMap(addr2, sizeof(data), false, true, true).Success());
    file2.Close();
    *((size_t*)addr1) = data;
    ASSERT_EQ(*(size_t*)addr2, data);
    file1.ShmUnlink();
    file2.ShmUnlink();
}
