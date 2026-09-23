/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#include <algorithm>
#include <fcntl.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include "detail/path_base.h"
#include "detail/types_helper.h"
#include "posix/cc/handle_cache.h"

class UCPosixHandleCacheTest : public UC::Test::Detail::PathBase {};

namespace {

std::string MakeFile(const std::string& dir, const char* name, char fill = 'A')
{
    auto path = dir + name;
    auto fd = ::open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    EXPECT_GE(fd, 0);
    char buf[64];
    std::fill(std::begin(buf), std::end(buf), fill);
    EXPECT_EQ(::write(fd, buf, sizeof(buf)), static_cast<ssize_t>(sizeof(buf)));
    ::close(fd);
    return path;
}

bool FdAlive(int32_t fd) { return fd >= 0 && ::fcntl(fd, F_GETFD) != -1; }

UC::Detail::BlockId MakeId(uint8_t tag)
{
    UC::Detail::BlockId id{};
    id[15] = static_cast<std::byte>(tag);
    return id;
}

}  // namespace

TEST_F(UCPosixHandleCacheTest, RepeatGetHitsSameFd)
{
    using namespace UC::PosixStore;
    LoadHandleCache cache;
    ASSERT_EQ(cache.Setup(8, false), UC::Status::OK());
    auto id = MakeId(1);
    auto path = MakeFile(Path(), "a.bin");
    int32_t firstFd = -1;
    {
        auto a = cache.GetOrOpen(id, path);
        ASSERT_TRUE(a.Valid());
        EXPECT_TRUE(a.Cached());
        firstFd = a.Fd();
    }
    auto b = cache.GetOrOpen(id, path);
    ASSERT_TRUE(b.Valid());
    EXPECT_TRUE(b.Cached());
    EXPECT_EQ(b.Fd(), firstFd);
}

TEST_F(UCPosixHandleCacheTest, CapacityZeroAlwaysUncached)
{
    using namespace UC::PosixStore;
    LoadHandleCache cache;
    ASSERT_EQ(cache.Setup(0, false), UC::Status::OK());
    auto id = MakeId(2);
    auto path = MakeFile(Path(), "b.bin");
    auto a = cache.GetOrOpen(id, path);
    ASSERT_TRUE(a.Valid());
    EXPECT_FALSE(a.Cached());
    auto first = a.Fd();
    a = {};
    EXPECT_FALSE(FdAlive(first));
    auto b = cache.GetOrOpen(id, path);
    ASSERT_TRUE(b.Valid());
    EXPECT_FALSE(b.Cached());
}

TEST_F(UCPosixHandleCacheTest, ClockEvictsWhenFull)
{
    using namespace UC::PosixStore;
    LoadHandleCache cache;
    ASSERT_EQ(cache.Setup(2, false), UC::Status::OK());
    auto id0 = MakeId(10);
    auto id1 = MakeId(11);
    auto id2 = MakeId(12);
    auto p0 = MakeFile(Path(), "c0.bin", '0');
    auto p1 = MakeFile(Path(), "c1.bin", '1');
    auto p2 = MakeFile(Path(), "c2.bin", '2');
    int32_t fd0 = -1;
    int32_t fd1 = -1;
    {
        auto a = cache.GetOrOpen(id0, p0);
        auto b = cache.GetOrOpen(id1, p1);
        ASSERT_TRUE(a.Valid());
        ASSERT_TRUE(b.Valid());
        fd0 = a.Fd();
        fd1 = b.Fd();
    }
    EXPECT_TRUE(FdAlive(fd0));
    EXPECT_TRUE(FdAlive(fd1));
    {
        auto c = cache.GetOrOpen(id2, p2);
        ASSERT_TRUE(c.Valid());
        EXPECT_TRUE(c.Cached());
    }
    EXPECT_FALSE(FdAlive(fd0) && FdAlive(fd1));
    auto again = cache.GetOrOpen(id2, p2);
    ASSERT_TRUE(again.Valid());
    EXPECT_TRUE(again.Cached());
}

TEST_F(UCPosixHandleCacheTest, PinProtectsFdFromEvict)
{
    using namespace UC::PosixStore;
    LoadHandleCache cache;
    ASSERT_EQ(cache.Setup(1, false), UC::Status::OK());
    auto id0 = MakeId(20);
    auto id1 = MakeId(21);
    auto p0 = MakeFile(Path(), "p0.bin", 'X');
    auto p1 = MakeFile(Path(), "p1.bin", 'Y');
    auto held = cache.GetOrOpen(id0, p0);
    ASSERT_TRUE(held.Valid());
    auto heldFd = held.Fd();
    ASSERT_TRUE(FdAlive(heldFd));
    auto other = cache.GetOrOpen(id1, p1);
    ASSERT_TRUE(other.Valid());
    EXPECT_TRUE(FdAlive(heldFd));
    char ch = 0;
    EXPECT_EQ(::pread(heldFd, &ch, 1, 0), 1);
    EXPECT_EQ(ch, 'X');
}

TEST_F(UCPosixHandleCacheTest, ConcurrentGetKeepsFdReadable)
{
    using namespace UC::PosixStore;
    LoadHandleCache cache;
    ASSERT_EQ(cache.Setup(4, false), UC::Status::OK());
    auto id = MakeId(50);
    auto path = MakeFile(Path(), "conc.bin", 'C');
    std::vector<std::thread> workers;
    std::atomic<int> ok{0};
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([&] {
            auto borrowed = cache.GetOrOpen(id, path);
            ASSERT_TRUE(borrowed.Valid());
            char ch = 0;
            EXPECT_EQ(::pread(borrowed.Fd(), &ch, 1, 0), 1);
            EXPECT_EQ(ch, 'C');
            ok.fetch_add(1);
        });
    }
    for (auto& th : workers) { th.join(); }
    EXPECT_EQ(ok.load(), 8);
}

TEST_F(UCPosixHandleCacheTest, MissingFileReturnsError)
{
    using namespace UC::PosixStore;
    LoadHandleCache cache;
    ASSERT_EQ(cache.Setup(4, false), UC::Status::OK());
    auto id = MakeId(60);
    auto borrowed = cache.GetOrOpen(id, Path() + "no-such.bin");
    EXPECT_FALSE(borrowed.Valid());
    EXPECT_EQ(borrowed.Error(), ENOENT);
}

TEST_F(UCPosixHandleCacheTest, PinnedSlotSurvivesFullSweep)
{
    using namespace UC::PosixStore;
    LoadHandleCache cache;
    ASSERT_EQ(cache.Setup(2, false), UC::Status::OK());
    auto held = cache.GetOrOpen(MakeId(80), MakeFile(Path(), "cas0.bin", 'A'));
    ASSERT_TRUE(held.Valid());
    auto heldFd = held.Fd();
    ASSERT_TRUE(FdAlive(heldFd));
    for (uint8_t i = 0; i < 6; ++i) {
        auto name = std::string("cas") + static_cast<char>('1' + i) + ".bin";
        auto other = cache.GetOrOpen(MakeId(90 + i), MakeFile(Path(), name.c_str(), 'B'));
        ASSERT_TRUE(other.Valid());
    }
    EXPECT_TRUE(FdAlive(heldFd));
    char ch = 0;
    EXPECT_EQ(::pread(heldFd, &ch, 1, 0), 1);
    EXPECT_EQ(ch, 'A');
}

TEST_F(UCPosixHandleCacheTest, CloseAllReleasesLiveFd)
{
    using namespace UC::PosixStore;
    LoadHandleCache cache;
    ASSERT_EQ(cache.Setup(4, false), UC::Status::OK());
    auto id = MakeId(70);
    auto path = MakeFile(Path(), "close.bin", 'K');
    int32_t fd = -1;
    {
        auto a = cache.GetOrOpen(id, path);
        ASSERT_TRUE(a.Valid());
        fd = a.Fd();
    }
    EXPECT_TRUE(FdAlive(fd));
    cache.CloseAll();
    EXPECT_FALSE(FdAlive(fd));
}

TEST_F(UCPosixHandleCacheTest, EvictedSlotIsReopenedOnNextGet)
{
    using namespace UC::PosixStore;
    LoadHandleCache cache;
    ASSERT_EQ(cache.Setup(1, false), UC::Status::OK());
    auto id0 = MakeId(40);
    auto id1 = MakeId(41);
    auto p0 = MakeFile(Path(), "t0.bin", '0');
    auto p1 = MakeFile(Path(), "t1.bin", '1');
    int32_t firstFd = -1;
    {
        auto a = cache.GetOrOpen(id0, p0);
        ASSERT_TRUE(a.Valid());
        ASSERT_TRUE(a.Cached());
        firstFd = a.Fd();
    }
    EXPECT_TRUE(FdAlive(firstFd));
    {
        auto b = cache.GetOrOpen(id1, p1);
        ASSERT_TRUE(b.Valid());
        ASSERT_TRUE(b.Cached());
    }
    EXPECT_FALSE(FdAlive(firstFd));
    auto again = cache.GetOrOpen(id0, p0);
    ASSERT_TRUE(again.Valid());
    EXPECT_TRUE(again.Cached());
    char ch = 0;
    EXPECT_EQ(::pread(again.Fd(), &ch, 1, 0), 1);
    EXPECT_EQ(ch, '0');
}
