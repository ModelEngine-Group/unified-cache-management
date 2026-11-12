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
#include "cache/shm_layout.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include "status/status.h"

namespace UC::Store {
namespace {

/** Reads the system-wide number of huge pages, 0 when unknown. */
std::size_t HugePagesTotal()
{
    const int fd = ::open("/proc/meminfo", O_RDONLY | O_CLOEXEC);
    if (fd < 0) { return 0; }
    std::string text;
    char chunk[512];
    for (;;) {
        const ssize_t n = ::read(fd, chunk, sizeof(chunk));
        if (n <= 0) { break; }
        text.append(chunk, static_cast<std::size_t>(n));
    }
    ::close(fd);

    const std::string key = "HugePages_Total:";
    const std::size_t at = text.find(key);
    if (at == std::string::npos) { return 0; }
    return static_cast<std::size_t>(std::strtoull(text.c_str() + at + key.size(), nullptr, 10));
}

TEST(ShmLayoutTest, GeometryRoundsUpToHugePageMultiple)
{
    // 3 x 3MB of slots = 9MB raw, rounded up to 10MB.
    auto total = ShmTotalSize(3, std::uint64_t{3} << 20);
    ASSERT_TRUE(total.HasValue()) << total.Error().ToString();
    EXPECT_EQ(total.Value(), std::size_t{10} << 20);

    // An exact multiple stays untouched: 1 x 2MB of slots.
    total = ShmTotalSize(1, std::uint64_t{2} << 20);
    ASSERT_TRUE(total.HasValue());
    EXPECT_EQ(total.Value(), std::size_t{2} << 20);
}

TEST(ShmLayoutTest, GeometryRejectsDegenerateAndOverflow)
{
    // Zero capacity and zero block size describe no pool at all.
    auto total = ShmTotalSize(0, 4096);
    ASSERT_FALSE(total.HasValue());
    EXPECT_EQ(total.Error(), Status::InvalidParam());

    total = ShmTotalSize(4, 0);
    ASSERT_FALSE(total.HasValue());
    EXPECT_EQ(total.Error(), Status::InvalidParam());

    // Four blocks of 2^62 bytes each overflow the 64-bit byte range.
    total = ShmTotalSize(4, std::uint64_t{1} << 62);
    ASSERT_FALSE(total.HasValue());
    EXPECT_EQ(total.Error(), Status::InvalidParam());
}

TEST(ShmLayoutTest, SlotAddressCrossesHugePageBoundary)
{
    const std::uint64_t blockSize = std::uint64_t{3} << 20;
    std::vector<std::uint8_t> fakeBase(3 * blockSize);
    void* base = fakeBase.data();
    EXPECT_EQ(SlotAddress(base, 0, blockSize), base);
    EXPECT_EQ(SlotAddress(base, 1, blockSize),
              fakeBase.data() + (std::size_t{3} << 20));  // crosses the 2MB line
    EXPECT_EQ(SlotAddress(base, 2, blockSize),
              fakeBase.data() + (std::size_t{6} << 20));  // crosses the 4MB line
}

TEST(ShmLayoutTest, TokenPayloadIsPureSlotTranslation)
{
    // Three slots encode to exactly twelve bytes, four per slot.
    const std::vector<std::uint32_t> slots = {0, 3, 7};
    const std::vector<std::byte> encoded = EncodeTokenPayload(slots);
    ASSERT_EQ(encoded.size(), slots.size() * 4);
    EXPECT_EQ(DecodeTokenPayload(encoded), slots);

    // An empty slot list round-trips as an empty payload.
    const std::vector<std::byte> empty = EncodeTokenPayload({});
    EXPECT_TRUE(empty.empty());
    EXPECT_TRUE(DecodeTokenPayload(empty).empty());

    // Pure translation: a tampered slot byte decodes to the tampered value.
    std::vector<std::byte> tampered = encoded;
    tampered[0] = std::byte{9};
    EXPECT_EQ(DecodeTokenPayload(tampered), (std::vector<std::uint32_t>{9, 3, 7}));

    // One to three trailing bytes are ignored: the count is the payload
    // length divided by four, rounded down.
    tampered.resize(tampered.size() + 2, std::byte{0xFF});
    EXPECT_EQ(DecodeTokenPayload(tampered), (std::vector<std::uint32_t>{9, 3, 7}));

    // Four appended bytes read back as one more slot: the length is the count.
    std::vector<std::byte> extended = encoded;
    extended.resize(extended.size() + 4, std::byte{0});
    EXPECT_EQ(DecodeTokenPayload(extended), (std::vector<std::uint32_t>{0, 3, 7, 0}));
}

TEST(ShmLayoutTest, OrchestratorAdaptsToHugePagePool)
{
    const bool poolPresent = HugePagesTotal() > 0;
    auto region = MemfdRegion::Create(2, std::uint64_t{1} << 20, true, "gtest-huge");
    if (poolPresent) {
        ASSERT_TRUE(region.HasValue()) << region.Error().ToString();
        EXPECT_TRUE(region.Value().HugePage());
        EXPECT_EQ(region.Value().Mapping().Size(), std::size_t{2} << 20);
    } else {
        ASSERT_FALSE(region.HasValue());
        EXPECT_EQ(region.Error(), Status::OsApiError());
    }
}

TEST(ShmLayoutTest, OrchestratorSucceedsWithoutForcingHugePages)
{
    auto region = MemfdRegion::Create(2, std::uint64_t{1} << 20, false, "gtest-soft");
    ASSERT_TRUE(region.HasValue()) << region.Error().ToString();
    EXPECT_EQ(region.Value().Mapping().Size(), std::size_t{2} << 20);
    // The page walk left its mark on the first byte of the region.
    EXPECT_EQ(*static_cast<const std::uint8_t*>(region.Value().Mapping().Base()), 1);
}

TEST(ShmLayoutTest, MappingAttachValidatesSize)
{
    const std::size_t size = std::size_t{3} << 20;
    const int fd = ::memfd_create("gtest-attach", MFD_CLOEXEC);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(::ftruncate(fd, static_cast<off_t>(size)), 0);

    auto attached = ShmMapping::Attach(fd, size);
    ASSERT_TRUE(attached.HasValue()) << attached.Error().ToString();
    EXPECT_NE(attached.Value().Base(), nullptr);
    EXPECT_EQ(attached.Value().Size(), size);
    attached.Value().Reset();
    EXPECT_FALSE(attached.Value().IsValid());

    auto mismatched = ShmMapping::Attach(fd, size + 1);
    ASSERT_FALSE(mismatched.HasValue());
    EXPECT_EQ(mismatched.Error(), Status::InvalidParam());

    auto badFd = ShmMapping::Attach(-1, size);
    ASSERT_FALSE(badFd.HasValue());
    EXPECT_EQ(badFd.Error(), Status::OsApiError());

    ::close(fd);
}

TEST(ShmLayoutTest, TouchedRegionIsFullyWritable)
{
    auto region = MemfdRegion::Create(2, std::uint64_t{1} << 20, false, "gtest-touch");
    ASSERT_TRUE(region.HasValue()) << region.Error().ToString();
    void* base = region.Value().Mapping().Base();

    // Fill both slots up to and including their last byte: a mapping shorter
    // than the geometry, or a lazy region, would SIGBUS here.
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
        auto* begin = static_cast<std::uint8_t*>(SlotAddress(base, slot, std::uint64_t{1} << 20));
        const std::uint8_t fill = static_cast<std::uint8_t>(0xA5 + slot);
        std::memset(begin, fill, std::size_t{1} << 20);
        EXPECT_EQ(begin[0], fill);
        EXPECT_EQ(begin[(std::size_t{1} << 20) - 1], fill);
    }
}

}  // namespace
}  // namespace UC::Store
