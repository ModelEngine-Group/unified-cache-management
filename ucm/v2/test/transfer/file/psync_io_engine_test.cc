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
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#include "file/io/io_engine.h"
#include "file/io/io_engine_registry.h"
#include "status/status.h"

namespace UC::Transfer {
namespace {

constexpr std::size_t kBlockSize = 4096;
constexpr std::size_t kIoDepth = 4;

std::vector<std::uint8_t> MakePattern(std::size_t size, std::uint8_t seed)
{
    std::vector<std::uint8_t> bytes(size);
    for (std::size_t i = 0; i < size; ++i) { bytes[i] = static_cast<std::uint8_t>(seed + i); }
    return bytes;
}

int WriteAt(int fd, const void* data, std::size_t size, std::size_t offset)
{
    const char* const cursor = static_cast<const char*>(data);
    std::size_t done = 0;
    while (done < size) {
        const ssize_t moved =
            ::pwrite(fd, cursor + done, size - done, static_cast<off_t>(offset + done));
        if (moved <= 0) { return -1; }
        done += static_cast<std::size_t>(moved);
    }
    return 0;
}

int ReadAt(int fd, void* data, std::size_t size, std::size_t offset)
{
    char* const cursor = static_cast<char*>(data);
    std::size_t done = 0;
    while (done < size) {
        const ssize_t moved =
            ::pread(fd, cursor + done, size - done, static_cast<off_t>(offset + done));
        if (moved <= 0) { return -1; }
        done += static_cast<std::size_t>(moved);
    }
    return 0;
}

IoEngine::IoJob MakeIoJob(std::vector<int32_t> fileHandles, std::vector<std::size_t> offsets,
                          std::vector<std::size_t> lengths, std::vector<void*> addresses)
{
    IoEngine::IoJob job;
    job.fileHandles = std::move(fileHandles);
    job.offsets = std::move(offsets);
    job.lengths = std::move(lengths);
    job.addresses = std::move(addresses);
    return job;
}

class TempFile {
public:
    TempFile()
    {
        path_ = ::testing::TempDir() + "ucm_psync_io_engine_test_XXXXXX";
        path_.push_back('\0');
        fd_ = ::mkstemp(path_.data());
    }
    ~TempFile()
    {
        if (fd_ >= 0) {
            ::close(fd_);
            ::remove(path_.c_str());
        }
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    int Fd() const { return fd_; }

private:
    std::string path_;
    int fd_ = -1;
};

class PsyncIoEngineTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        auto created = IoEngineRegistry::Instance().Create(IoEngineKind::Psync);
        ASSERT_TRUE(created.HasValue()) << created.Error().ToString();
        engine_.reset(*created);
        const IoEngine::InitParam param{kIoDepth};
        ASSERT_TRUE(engine_->Setup(param).Success());
    }

    void TearDown() override { engine_.reset(); }

    std::vector<std::pair<std::size_t, bool>> DrainFinished(std::size_t expectedCount)
    {
        std::vector<std::pair<std::size_t, bool>> collected;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (collected.size() < expectedCount && std::chrono::steady_clock::now() < deadline) {
            auto finished = engine_->GetFinished();
            if (finished.HasValue()) {
                collected.insert(collected.end(), finished.Value().begin(), finished.Value().end());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return collected;
    }

    std::unique_ptr<IoEngine> engine_;
};

TEST(PsyncIoEngineSetupTest, SetupRejectsZeroIoDepth)
{
    auto created = IoEngineRegistry::Instance().Create(IoEngineKind::Psync);
    ASSERT_TRUE(created.HasValue());
    std::unique_ptr<IoEngine> engine(*created);
    const IoEngine::InitParam param{0};
    EXPECT_EQ(engine->Setup(param), Status::InvalidParam());
}

TEST_F(PsyncIoEngineTest, MalformedJobsAreRejectedWithoutOccupyingTheJobId)
{
    TempFile file;
    ASSERT_GE(file.Fd(), 0);
    std::vector<std::uint8_t> buffer(kBlockSize);

    EXPECT_EQ(engine_->SubmitReadV(1, IoEngine::IoJob{}), Status::InvalidParam());
    EXPECT_EQ(engine_->SubmitReadV(
                  1, MakeIoJob({file.Fd(), file.Fd()}, {0, 0}, {kBlockSize}, {buffer.data()})),
              Status::InvalidParam());
    EXPECT_EQ(engine_->SubmitReadV(1, MakeIoJob({file.Fd()}, {0}, {kBlockSize}, {nullptr})),
              Status::InvalidParam());
    EXPECT_EQ(engine_->SubmitReadV(1, MakeIoJob({file.Fd()}, {0}, {0}, {buffer.data()})),
              Status::InvalidParam());

    EXPECT_TRUE(engine_->SubmitWriteV(1, MakeIoJob({file.Fd()}, {0}, {kBlockSize}, {buffer.data()}))
                    .Success());
    EXPECT_EQ(engine_->Wait(1), Status::Ok());
}

TEST_F(PsyncIoEngineTest, DuplicateJobIdIsRejectedWhileInFlightAndReusableAfterConsumption)
{
    TempFile file;
    ASSERT_GE(file.Fd(), 0);
    const auto pattern = MakePattern(kBlockSize, 0x10);
    ASSERT_EQ(WriteAt(file.Fd(), pattern.data(), pattern.size(), 0), 0);

    std::vector<std::uint8_t> buffer(kBlockSize);
    EXPECT_TRUE(engine_->SubmitReadV(7, MakeIoJob({file.Fd()}, {0}, {kBlockSize}, {buffer.data()}))
                    .Success());
    EXPECT_EQ(engine_->SubmitReadV(7, MakeIoJob({file.Fd()}, {0}, {kBlockSize}, {buffer.data()})),
              Status::DuplicateKey());

    EXPECT_EQ(engine_->Wait(7), Status::Ok());
    EXPECT_EQ(std::memcmp(buffer.data(), pattern.data(), pattern.size()), 0);

    EXPECT_TRUE(engine_->SubmitReadV(7, MakeIoJob({file.Fd()}, {0}, {kBlockSize}, {buffer.data()}))
                    .Success());
    EXPECT_EQ(engine_->Wait(7), Status::Ok());
}

TEST_F(PsyncIoEngineTest, ReadRoundTripAcrossSegments)
{
    TempFile file;
    ASSERT_GE(file.Fd(), 0);
    const auto first = MakePattern(kBlockSize, 0x00);
    const auto second = MakePattern(kBlockSize / 2, 0x40);
    const auto third = MakePattern(kBlockSize, 0x80);
    ASSERT_EQ(WriteAt(file.Fd(), first.data(), first.size(), 0), 0);
    ASSERT_EQ(WriteAt(file.Fd(), second.data(), second.size(), kBlockSize), 0);
    ASSERT_EQ(WriteAt(file.Fd(), third.data(), third.size(), kBlockSize + kBlockSize / 2), 0);

    std::vector<std::uint8_t> bufferFirst(kBlockSize);
    std::vector<std::uint8_t> bufferSecond(kBlockSize / 2);
    std::vector<std::uint8_t> bufferThird(kBlockSize);
    EXPECT_TRUE(engine_
                    ->SubmitReadV(
                        1, MakeIoJob({file.Fd(), file.Fd(), file.Fd()},
                                     {0, kBlockSize, kBlockSize + kBlockSize / 2},
                                     {kBlockSize, kBlockSize / 2, kBlockSize},
                                     {bufferFirst.data(), bufferSecond.data(), bufferThird.data()}))
                    .Success());
    EXPECT_EQ(engine_->Wait(1), Status::Ok());
    EXPECT_EQ(std::memcmp(bufferFirst.data(), first.data(), first.size()), 0);
    EXPECT_EQ(std::memcmp(bufferSecond.data(), second.data(), second.size()), 0);
    EXPECT_EQ(std::memcmp(bufferThird.data(), third.data(), third.size()), 0);
}

TEST_F(PsyncIoEngineTest, WriteRoundTripAcrossSegments)
{
    TempFile file;
    ASSERT_GE(file.Fd(), 0);
    auto first = MakePattern(kBlockSize, 0x11);
    auto second = MakePattern(kBlockSize, 0x22);
    auto third = MakePattern(kBlockSize / 4, 0x33);

    EXPECT_TRUE(engine_
                    ->SubmitWriteV(3, MakeIoJob({file.Fd(), file.Fd(), file.Fd()},
                                                {kBlockSize, 0, 2 * kBlockSize},
                                                {kBlockSize, kBlockSize, kBlockSize / 4},
                                                {first.data(), second.data(), third.data()}))
                    .Success());
    EXPECT_EQ(engine_->Wait(3), Status::Ok());

    std::vector<std::uint8_t> check(2 * kBlockSize + kBlockSize / 4);
    ASSERT_EQ(ReadAt(file.Fd(), check.data(), check.size(), 0), 0);
    EXPECT_EQ(std::memcmp(check.data(), second.data(), kBlockSize), 0);
    EXPECT_EQ(std::memcmp(check.data() + kBlockSize, first.data(), kBlockSize), 0);
    EXPECT_EQ(std::memcmp(check.data() + 2 * kBlockSize, third.data(), kBlockSize / 4), 0);
}

TEST_F(PsyncIoEngineTest, GetFinishedReportsEachCompletionExactlyOnce)
{
    TempFile file;
    ASSERT_GE(file.Fd(), 0);
    const auto pattern = MakePattern(kBlockSize, 0x55);
    ASSERT_EQ(WriteAt(file.Fd(), pattern.data(), pattern.size(), 0), 0);

    constexpr std::size_t kJobCount = 8;
    std::vector<std::uint8_t> buffer(kBlockSize);
    for (std::size_t id = 1; id <= kJobCount; ++id) {
        EXPECT_TRUE(
            engine_->SubmitReadV(id, MakeIoJob({file.Fd()}, {0}, {kBlockSize}, {buffer.data()}))
                .Success());
    }

    const auto collected = DrainFinished(kJobCount);
    ASSERT_EQ(collected.size(), kJobCount);
    for (std::size_t id = 1; id <= kJobCount; ++id) {
        const auto found = std::find_if(
            collected.begin(), collected.end(),
            [id](const std::pair<std::size_t, bool>& entry) { return entry.first == id; });
        ASSERT_TRUE(found != collected.end());
        EXPECT_TRUE(found->second);
    }

    auto again = engine_->GetFinished();
    ASSERT_TRUE(again.HasValue());
    EXPECT_TRUE(again.Value().empty());

    EXPECT_TRUE(
        engine_
            ->SubmitReadV(kJobCount + 1, MakeIoJob({file.Fd()}, {0}, {kBlockSize}, {buffer.data()}))
            .Success());
    EXPECT_EQ(engine_->Wait(kJobCount + 1), Status::Ok());
    auto drained = engine_->GetFinished();
    ASSERT_TRUE(drained.HasValue());
    EXPECT_TRUE(drained.Value().empty());
}

TEST_F(PsyncIoEngineTest, WaitOnlyUsageKeepsExactlyOnceAcrossIdReuse)
{
    TempFile file;
    ASSERT_GE(file.Fd(), 0);
    const auto pattern = MakePattern(kBlockSize, 0x99);
    ASSERT_EQ(WriteAt(file.Fd(), pattern.data(), pattern.size(), 0), 0);

    constexpr std::size_t kJobCount = 8;
    std::vector<std::uint8_t> buffer(kBlockSize);
    for (std::size_t id = 1; id <= kJobCount; ++id) {
        ASSERT_TRUE(
            engine_->SubmitReadV(id, MakeIoJob({file.Fd()}, {0}, {kBlockSize}, {buffer.data()}))
                .Success());
        EXPECT_EQ(engine_->Wait(id), Status::Ok());
    }

    auto stale = engine_->GetFinished();
    ASSERT_TRUE(stale.HasValue());
    EXPECT_TRUE(stale.Value().empty());

    for (std::size_t id = 1; id <= kJobCount; ++id) {
        ASSERT_TRUE(
            engine_->SubmitReadV(id, MakeIoJob({file.Fd()}, {0}, {kBlockSize}, {buffer.data()}))
                .Success());
    }
    const auto collected = DrainFinished(kJobCount);
    ASSERT_EQ(collected.size(), kJobCount);
    for (std::size_t id = 1; id <= kJobCount; ++id) {
        const auto found = std::find_if(
            collected.begin(), collected.end(),
            [id](const std::pair<std::size_t, bool>& entry) { return entry.first == id; });
        ASSERT_TRUE(found != collected.end());
        EXPECT_TRUE(found->second);
    }
    auto again = engine_->GetFinished();
    ASSERT_TRUE(again.HasValue());
    EXPECT_TRUE(again.Value().empty());
}

TEST_F(PsyncIoEngineTest, WaitRejectsUnknownAndConsumedJobIds)
{
    TempFile file;
    ASSERT_GE(file.Fd(), 0);
    const auto pattern = MakePattern(kBlockSize, 0x66);
    ASSERT_EQ(WriteAt(file.Fd(), pattern.data(), pattern.size(), 0), 0);

    std::vector<std::uint8_t> buffer(kBlockSize);
    EXPECT_EQ(engine_->Wait(42), Status::InvalidParam());
    EXPECT_TRUE(engine_->SubmitReadV(1, MakeIoJob({file.Fd()}, {0}, {kBlockSize}, {buffer.data()}))
                    .Success());
    EXPECT_EQ(engine_->Wait(1), Status::Ok());
    EXPECT_EQ(engine_->Wait(1), Status::InvalidParam());
}

TEST_F(PsyncIoEngineTest, ReadPastEndOfFileFailsTheJob)
{
    TempFile file;
    ASSERT_GE(file.Fd(), 0);
    const auto pattern = MakePattern(kBlockSize, 0x77);
    ASSERT_EQ(WriteAt(file.Fd(), pattern.data(), pattern.size(), 0), 0);

    std::vector<std::uint8_t> buffer(2 * kBlockSize);
    EXPECT_TRUE(
        engine_->SubmitReadV(1, MakeIoJob({file.Fd()}, {0}, {buffer.size()}, {buffer.data()}))
            .Success());
    EXPECT_EQ(engine_->Wait(1), Status::OsApiError());

    EXPECT_TRUE(
        engine_->SubmitReadV(2, MakeIoJob({file.Fd()}, {kBlockSize}, {kBlockSize}, {buffer.data()}))
            .Success());
    const auto collected = DrainFinished(1);
    ASSERT_EQ(collected.size(), std::size_t{1});
    EXPECT_EQ(collected[0].first, std::size_t{2});
    EXPECT_FALSE(collected[0].second);
}

TEST_F(PsyncIoEngineTest, BadFileDescriptorFailsTheJob)
{
    std::vector<std::uint8_t> buffer(kBlockSize);
    EXPECT_TRUE(
        engine_->SubmitReadV(1, MakeIoJob({-1}, {0}, {kBlockSize}, {buffer.data()})).Success());
    EXPECT_EQ(engine_->Wait(1), Status::OsApiError());
}

TEST(PsyncIoEngineFailureTest, FirstSegmentFailureCancelsUnstartedSegments)
{
    auto created = IoEngineRegistry::Instance().Create(IoEngineKind::Psync);
    ASSERT_TRUE(created.HasValue());
    std::unique_ptr<IoEngine> engine(*created);
    // A single worker keeps segment execution strictly in queue order, so the
    // cancellation of the trailing segments is observable exactly.
    const IoEngine::InitParam param{1};
    ASSERT_TRUE(engine->Setup(param).Success());

    TempFile file;
    ASSERT_GE(file.Fd(), 0);
    constexpr std::size_t kSegmentCount = 16;
    std::array<std::vector<std::uint8_t>, kSegmentCount> patterns;
    for (std::size_t segment = 0; segment < kSegmentCount; ++segment) {
        patterns[segment] = MakePattern(kBlockSize, static_cast<std::uint8_t>(0xa0 + segment));
    }

    std::vector<int32_t> fds;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> lengths;
    std::vector<void*> addresses;
    fds.push_back(-1);  // the first segment fails instantly, cancelling the rest
    offsets.push_back(0);
    lengths.push_back(kBlockSize);
    addresses.push_back(patterns[0].data());
    for (std::size_t segment = 1; segment < kSegmentCount; ++segment) {
        fds.push_back(file.Fd());
        offsets.push_back(segment * kBlockSize);
        lengths.push_back(kBlockSize);
        addresses.push_back(patterns[segment].data());
    }

    ASSERT_TRUE(engine->SubmitWriteV(1, MakeIoJob(fds, offsets, lengths, addresses)).Success());
    EXPECT_EQ(engine->Wait(1), Status::OsApiError());
    EXPECT_EQ(::lseek(file.Fd(), 0, SEEK_END), 0);  // no trailing segment ever wrote
}

TEST_F(PsyncIoEngineTest, FailureWithConcurrentSiblingsSettlesExactlyOnce)
{
    TempFile file;
    ASSERT_GE(file.Fd(), 0);
    const auto pattern = MakePattern(kBlockSize, 0x88);
    ASSERT_EQ(WriteAt(file.Fd(), pattern.data(), pattern.size(), 0), 0);

    constexpr std::size_t kSegmentCount = 32;
    std::vector<std::uint8_t> buffer(kBlockSize);
    std::vector<int32_t> fds;
    std::vector<std::size_t> offsets(kSegmentCount, 0);
    std::vector<std::size_t> lengths(kSegmentCount, kBlockSize);
    std::vector<void*> addresses(kSegmentCount, buffer.data());
    for (std::size_t segment = 0; segment < kSegmentCount; ++segment) {
        fds.push_back(segment == kSegmentCount / 2 ? -1 : file.Fd());
    }

    ASSERT_TRUE(engine_->SubmitReadV(1, MakeIoJob(fds, offsets, lengths, addresses)).Success());
    const auto collected = DrainFinished(1);
    ASSERT_EQ(collected.size(), std::size_t{1});
    EXPECT_EQ(collected[0].first, std::size_t{1});
    EXPECT_FALSE(collected[0].second);

    auto drained = engine_->GetFinished();
    ASSERT_TRUE(drained.HasValue());
    EXPECT_TRUE(drained.Value().empty());

    // the cancelled job must not corrupt the accounting: the id is reusable
    EXPECT_TRUE(engine_->SubmitReadV(1, MakeIoJob({file.Fd()}, {0}, {kBlockSize}, {buffer.data()}))
                    .Success());
    EXPECT_EQ(engine_->Wait(1), Status::Ok());
}

TEST_F(PsyncIoEngineTest, ConcurrentJobsWriteDisjointRegionsOfOneFile)
{
    TempFile file;
    ASSERT_GE(file.Fd(), 0);
    constexpr std::size_t kJobCount = 8;
    std::array<std::vector<std::uint8_t>, kJobCount> patterns;
    for (std::size_t job = 0; job < kJobCount; ++job) {
        patterns[job] = MakePattern(kBlockSize, static_cast<std::uint8_t>(job * 0x10));
    }
    for (std::size_t job = 0; job < kJobCount; ++job) {
        EXPECT_TRUE(engine_
                        ->SubmitWriteV(job + 1, MakeIoJob({file.Fd()}, {job * kBlockSize},
                                                          {kBlockSize}, {patterns[job].data()}))
                        .Success());
    }
    const auto collected = DrainFinished(kJobCount);
    ASSERT_EQ(collected.size(), kJobCount);
    for (const auto& entry : collected) { EXPECT_TRUE(entry.second); }

    std::vector<std::uint8_t> check(kJobCount * kBlockSize);
    ASSERT_EQ(ReadAt(file.Fd(), check.data(), check.size(), 0), 0);
    for (std::size_t job = 0; job < kJobCount; ++job) {
        EXPECT_EQ(std::memcmp(check.data() + job * kBlockSize, patterns[job].data(), kBlockSize),
                  0);
    }
}

}  // namespace
}  // namespace UC::Transfer
