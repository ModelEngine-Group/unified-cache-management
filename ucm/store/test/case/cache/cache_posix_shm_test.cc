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
#include <gtest/gtest.h>
#include "cache/cc/posix_shm.h"
#include "detail/data_generator.h"
#include "detail/random.h"

class UCCachePosixShmTest : public testing::Test {
public:
    UC::Test::Detail::Random rd;
};

TEST_F(UCCachePosixShmTest, ShmMapAndUnmap)
{
    auto fileName = rd.RandomString(20);
    UC::CacheStore::PosixShm file1{fileName};
    UC::CacheStore::PosixShm file2{fileName};
    const size_t data = 0xfffffffe;
    const auto openFlags =
        UC::CacheStore::PosixShm::OpenFlag::READ_WRITE | UC::CacheStore::PosixShm::OpenFlag::CREATE;
    void* addr1 = nullptr;
    void* addr2 = nullptr;
    ASSERT_TRUE(file1.ShmOpen(openFlags).Success());
    ASSERT_TRUE(file1.Truncate(sizeof(data)).Success());
    ASSERT_TRUE(file1.MMap(addr1, sizeof(data), true, true, true).Success());
    ASSERT_TRUE(file2.ShmOpen(openFlags).Success());
    ASSERT_TRUE(file2.MMap(addr2, sizeof(data), false, true, true).Success());
    file1.ShmUnlink();
    file2.ShmUnlink();
    *((size_t*)addr1) = data;
    ASSERT_EQ(*(size_t*)addr2, data);
    UC::CacheStore::PosixShm::MUnmap(addr1, sizeof(data));
    UC::CacheStore::PosixShm::MUnmap(addr2, sizeof(data));
}
