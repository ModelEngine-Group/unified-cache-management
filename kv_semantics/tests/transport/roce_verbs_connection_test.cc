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

// Unit tests of the CPU data path bookkeeping.  A fake queue pair stands in for
// the device, so the cases run without rdma-core: post results, the receive ring,
// cid matching of out of order answers, the copy-before-repost rule and the
// failure paths are all exercised here.

#include <cstdint>
#include <cstring>
#include <deque>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "kv_types.h"
#include "roce/roce_io_wire.h"
#include "roce/verbs_connection.h"

namespace kv::test {
namespace {

constexpr std::uint32_t kLkey = 0x1234U;
constexpr std::uint16_t kCid = 0x0042U;
constexpr std::size_t kResponseCapacity = 71U;

std::uint32_t StatusDword(std::uint16_t cid, std::uint16_t status)
{
    return (static_cast<std::uint32_t>(cid) & roce::kSqeCidMask) |
           ((static_cast<std::uint32_t>(status) & roce::kCqeStatusMask) << roce::kCqeStatusShift);
}

// Builds a completion record the store appliance would send: the sixteen byte
// header followed by the result buffer.
std::vector<std::uint8_t> MakeResponse(std::uint16_t cid, std::uint16_t status)
{
    std::vector<std::uint8_t> response(roce::kResponseMaxBytes, 0U);
    const std::uint32_t statusDword = StatusDword(cid, status);
    std::memcpy(response.data() + 12U, &statusDword, sizeof(statusDword));
    response[roce::kResponseHeaderBytes] = 0x0FU;  // first four bit result
    return response;
}

class FakeQueuePairIo final : public roce::VerbsConnection::QueuePairIo {
public:
    struct Send {
        const void* buffer;
        std::size_t length;
        std::uint32_t lkey;
        std::uint64_t wrId;
    };

    struct Receive {
        void* buffer;
        std::size_t length;
        std::uint32_t lkey;
        std::uint64_t wrId;
    };

    Status PostSend(const void* buffer, std::size_t length, std::uint32_t lkey,
                    std::uint64_t wrId, std::string& error) override
    {
        if (failPostSend) {
            error = "fake send failure";
            return Status::Error(StatusCode::IO_ERROR, error);
        }
        sends.push_back(Send{buffer, length, lkey, wrId});
        return Status::OK();
    }

    Status PostReceive(void* buffer, std::size_t length, std::uint32_t lkey, std::uint64_t wrId,
                       std::string& error) override
    {
        if (failPostReceive) {
            error = "fake receive failure";
            return Status::Error(StatusCode::IO_ERROR, error);
        }
        receives.push_back(Receive{buffer, length, lkey, wrId});
        return Status::OK();
    }

    std::size_t Poll(roce::Completion* completions, std::size_t maxCompletions) override
    {
        std::size_t count = 0;
        while (count < maxCompletions && !pending.empty()) {
            completions[count] = pending.front();
            pending.pop_front();
            ++count;
        }
        return count;
    }

    void CompleteReceive(std::uint32_t slot, std::uint16_t cid, std::uint16_t status,
                         std::size_t responseBytes = kResponseCapacity)
    {
        Receive& receive = receives.at(slot);
        const std::vector<std::uint8_t> response = MakeResponse(cid, status);
        const std::size_t copied = response.size() < responseBytes ? response.size() : responseBytes;
        std::memcpy(receive.buffer, response.data(), copied);
        roce::Completion completion;
        completion.wrId = static_cast<std::uint64_t>(slot) << 1U;
        completion.type = roce::CompletionType::Receive;
        completion.result = roce::CompletionResult::Success;
        completion.byteLength = static_cast<std::uint32_t>(copied);
        pending.push_back(completion);
    }

    void CompleteSend(std::uint64_t wrId, bool success)
    {
        roce::Completion completion;
        completion.wrId = wrId;
        completion.type = roce::CompletionType::Send;
        completion.result =
            success ? roce::CompletionResult::Success : roce::CompletionResult::Failed;
        pending.push_back(completion);
    }

    bool failPostSend{false};
    bool failPostReceive{false};
    std::vector<Send> sends;
    std::vector<Receive> receives;
    std::deque<roce::Completion> pending;
};

struct TestConnection {
    FakeQueuePairIo* io{nullptr};
    std::unique_ptr<roce::VerbsConnection> connection;
};

TestConnection MakeConnection(std::uint32_t receiveDepth)
{
    auto io = std::make_unique<FakeQueuePairIo>();
    TestConnection fixture;
    fixture.io = io.get();
    fixture.connection = roce::VerbsConnection::OpenForTest(std::move(io), receiveDepth, kLkey);
    return fixture;
}

std::vector<std::uint8_t> MakeFlagBuffer(std::size_t responseBytes, std::uint8_t fill = 0)
{
    return std::vector<std::uint8_t>(responseBytes, fill);
}

TEST(CpuKvOverRoceVerbsConnectionTest, PostsTheWholeReceiveRingOnOpen)
{
    TestConnection fixture = MakeConnection(4);
    ASSERT_NE(nullptr, fixture.connection);
    std::string error;
    ASSERT_TRUE(fixture.connection->PostReceiveRing(error).ok()) << error;

    ASSERT_EQ(4U, fixture.io->receives.size());
    for (std::uint32_t index = 0; index < 4U; ++index) {
        EXPECT_EQ(static_cast<std::uint64_t>(index) << 1U, fixture.io->receives[index].wrId);
        EXPECT_EQ(kLkey, fixture.io->receives[index].lkey);
        EXPECT_EQ(roce::kResponseRecordBytes, fixture.io->receives[index].length);
    }
    EXPECT_EQ(0U, fixture.io->sends.size());
    EXPECT_EQ(4U, fixture.connection->FreeReceiveSlots());
}

TEST(CpuKvOverRoceVerbsConnectionTest, PostRequestRegistersTheCidAndPostsTheSend)
{
    TestConnection fixture = MakeConnection(4);
    std::string error;
    ASSERT_TRUE(fixture.connection->PostReceiveRing(error).ok()) << error;

    std::uint32_t sqe[roce::kSqeHeaderDwords] = {};
    std::uint8_t flag[kResponseCapacity] = {};
    ASSERT_TRUE(fixture.connection
                    ->PostRequest(sqe, sizeof(sqe), kLkey, flag, kCid, kResponseCapacity, error)
                    .ok())
        << error;

    ASSERT_EQ(1U, fixture.io->sends.size());
    EXPECT_EQ(sqe, fixture.io->sends.front().buffer);
    EXPECT_EQ(sizeof(sqe), fixture.io->sends.front().length);
    EXPECT_EQ(kLkey, fixture.io->sends.front().lkey);
    EXPECT_EQ(3U, fixture.connection->FreeReceiveSlots());
}

TEST(CpuKvOverRoceVerbsConnectionTest, PostRequestReportsResourceBusyWhenTheRingIsFull)
{
    TestConnection fixture = MakeConnection(2);
    std::string error;
    ASSERT_TRUE(fixture.connection->PostReceiveRing(error).ok()) << error;

    std::uint32_t sqe[roce::kSqeHeaderDwords] = {};
    std::uint8_t flag[kResponseCapacity] = {};
    ASSERT_TRUE(fixture.connection
                    ->PostRequest(sqe, sizeof(sqe), kLkey, flag, 1U, kResponseCapacity, error)
                    .ok());
    ASSERT_TRUE(fixture.connection
                    ->PostRequest(sqe, sizeof(sqe), kLkey, flag, 2U, kResponseCapacity, error)
                    .ok());

    const Status third =
        fixture.connection->PostRequest(sqe, sizeof(sqe), kLkey, flag, 3U, kResponseCapacity,
                                        error);
    EXPECT_EQ(StatusCode::RESOURCE_BUSY, third.code) << third.message;
    EXPECT_EQ(2U, fixture.io->sends.size());
    EXPECT_EQ(0U, fixture.connection->FreeReceiveSlots());
}

TEST(CpuKvOverRoceVerbsConnectionTest, PostRequestReportsIoErrorWhenTheSendFails)
{
    TestConnection fixture = MakeConnection(2);
    std::string error;
    ASSERT_TRUE(fixture.connection->PostReceiveRing(error).ok()) << error;

    fixture.io->failPostSend = true;
    std::uint32_t sqe[roce::kSqeHeaderDwords] = {};
    std::uint8_t flag[kResponseCapacity] = {};
    const Status failed = fixture.connection->PostRequest(sqe, sizeof(sqe), kLkey, flag, 1U,
                                                          kResponseCapacity, error);
    EXPECT_EQ(StatusCode::IO_ERROR, failed.code) << failed.message;
    // The failed request must not keep a receive slot claimed.
    EXPECT_EQ(2U, fixture.connection->FreeReceiveSlots());
    EXPECT_FALSE(fixture.connection->Failed());

    fixture.io->failPostSend = false;
    ASSERT_TRUE(fixture.connection
                    ->PostRequest(sqe, sizeof(sqe), kLkey, flag, 2U, kResponseCapacity, error)
                    .ok());
}

TEST(CpuKvOverRoceVerbsConnectionTest, PostRequestRejectsUnusableArguments)
{
    TestConnection fixture = MakeConnection(2);
    std::string error;
    ASSERT_TRUE(fixture.connection->PostReceiveRing(error).ok()) << error;

    std::uint32_t sqe[roce::kSqeHeaderDwords] = {};
    std::uint8_t flag[kResponseCapacity] = {};
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
              fixture.connection->PostRequest(nullptr, sizeof(sqe), kLkey, flag, kCid,
                                              kResponseCapacity, error)
                  .code);
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
              fixture.connection->PostRequest(sqe, 0U, kLkey, flag, kCid, kResponseCapacity,
                                              error)
                  .code);
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
              fixture.connection->PostRequest(sqe, sizeof(sqe), kLkey, nullptr, kCid,
                                              kResponseCapacity, error)
                  .code);
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
              fixture.connection->PostRequest(sqe, sizeof(sqe), kLkey, flag, 0U, kResponseCapacity,
                                              error)
                  .code);
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
              fixture.connection->PostRequest(sqe, sizeof(sqe), kLkey, flag, kCid,
                                              roce::kResponseHeaderBytes - 1U, error)
                  .code);
    EXPECT_TRUE(fixture.io->sends.empty());
}

TEST(CpuKvOverRoceVerbsConnectionTest, CompletionCopiesTheAnswerIntoTheFlagBufferBeforeReposting)
{
    TestConnection fixture = MakeConnection(2);
    std::string error;
    ASSERT_TRUE(fixture.connection->PostReceiveRing(error).ok()) << error;

    std::uint32_t sqe[roce::kSqeHeaderDwords] = {};
    std::vector<std::uint8_t> flag = MakeFlagBuffer(kResponseCapacity);
    ASSERT_TRUE(fixture.connection
                    ->PostRequest(sqe, sizeof(sqe), kLkey, flag.data(), kCid, kResponseCapacity,
                                  error)
                    .ok());

    const std::size_t receivesBefore = fixture.io->receives.size();
    fixture.io->CompleteReceive(0U, kCid, 0U);
    fixture.connection->PollCompletion();

    std::uint32_t publishedStatus = 0;
    std::memcpy(&publishedStatus, flag.data() + 12U, sizeof(publishedStatus));
    EXPECT_EQ(StatusDword(kCid, 0U), publishedStatus);
    EXPECT_EQ(0x0FU, flag[roce::kResponseHeaderBytes]);
    // The slot is reposted after the copy, with the same buffer.
    EXPECT_EQ(receivesBefore + 1U, fixture.io->receives.size());
    EXPECT_EQ(0U, fixture.io->receives.back().wrId);
    // The request was matched, so the finished slot is free again.
    EXPECT_EQ(2U, fixture.connection->FreeReceiveSlots());
}

TEST(CpuKvOverRoceVerbsConnectionTest, AnswersAreMatchedByCidNotByOrder)
{
    TestConnection fixture = MakeConnection(3);
    std::string error;
    ASSERT_TRUE(fixture.connection->PostReceiveRing(error).ok()) << error;

    std::uint32_t sqe[roce::kSqeHeaderDwords] = {};
    std::vector<std::uint8_t> firstFlag = MakeFlagBuffer(kResponseCapacity);
    std::vector<std::uint8_t> secondFlag = MakeFlagBuffer(kResponseCapacity);
    std::vector<std::uint8_t> thirdFlag = MakeFlagBuffer(kResponseCapacity);
    ASSERT_TRUE(fixture.connection
                    ->PostRequest(sqe, sizeof(sqe), kLkey, firstFlag.data(), 1U, kResponseCapacity,
                                  error)
                    .ok());
    ASSERT_TRUE(fixture.connection
                    ->PostRequest(sqe, sizeof(sqe), kLkey, secondFlag.data(), 2U, kResponseCapacity,
                                  error)
                    .ok());
    ASSERT_TRUE(fixture.connection
                    ->PostRequest(sqe, sizeof(sqe), kLkey, thirdFlag.data(), 3U, kResponseCapacity,
                                  error)
                    .ok());

    // The third request is answered first, the first one second.
    fixture.io->CompleteReceive(1U, 3U, 0U);
    fixture.io->CompleteReceive(0U, 1U, 0U);
    fixture.connection->PollCompletion();

    std::uint32_t published = 0;
    std::memcpy(&published, thirdFlag.data() + 12U, sizeof(published));
    EXPECT_EQ(StatusDword(3U, 0U), published);
    std::memcpy(&published, firstFlag.data() + 12U, sizeof(published));
    EXPECT_EQ(StatusDword(1U, 0U), published);
    // The second request is still waiting and its flag is untouched.
    std::memcpy(&published, secondFlag.data() + 12U, sizeof(published));
    EXPECT_EQ(0U, published);
    EXPECT_EQ(2U, fixture.connection->FreeReceiveSlots());
}

TEST(CpuKvOverRoceVerbsConnectionTest, AnAnswerWithoutAMatchingRequestIsDroppedAndReposted)
{
    TestConnection fixture = MakeConnection(2);
    std::string error;
    ASSERT_TRUE(fixture.connection->PostReceiveRing(error).ok()) << error;

    const std::size_t receivesBefore = fixture.io->receives.size();
    fixture.io->CompleteReceive(0U, 0x7777U, 0U);
    fixture.connection->PollCompletion();

    EXPECT_FALSE(fixture.connection->Failed());
    EXPECT_EQ(receivesBefore + 1U, fixture.io->receives.size());
    EXPECT_EQ(2U, fixture.connection->FreeReceiveSlots());
}

TEST(CpuKvOverRoceVerbsConnectionTest, RepostFailureMarksTheConnectionFailed)
{
    TestConnection fixture = MakeConnection(2);
    std::string error;
    ASSERT_TRUE(fixture.connection->PostReceiveRing(error).ok()) << error;

    std::uint32_t sqe[roce::kSqeHeaderDwords] = {};
    std::vector<std::uint8_t> flag = MakeFlagBuffer(kResponseCapacity);
    ASSERT_TRUE(fixture.connection
                    ->PostRequest(sqe, sizeof(sqe), kLkey, flag.data(), kCid, kResponseCapacity,
                                  error)
                    .ok());

    fixture.io->failPostReceive = true;
    fixture.io->CompleteReceive(0U, kCid, 0U);
    fixture.connection->PollCompletion();

    EXPECT_TRUE(fixture.connection->Failed());
    EXPECT_EQ(StatusCode::CONNECTION_ERROR,
              fixture.connection
                  ->PostRequest(sqe, sizeof(sqe), kLkey, flag.data(), kCid, kResponseCapacity,
                                error)
                  .code);
}

TEST(CpuKvOverRoceVerbsConnectionTest, AFailedCompletionMarksTheConnectionFailed)
{
    TestConnection fixture = MakeConnection(2);
    std::string error;
    ASSERT_TRUE(fixture.connection->PostReceiveRing(error).ok()) << error;

    roce::Completion completion;
    completion.wrId = 0U;
    completion.type = roce::CompletionType::Receive;
    completion.result = roce::CompletionResult::Failed;
    fixture.io->pending.push_back(completion);
    fixture.connection->PollCompletion();

    EXPECT_TRUE(fixture.connection->Failed());
}

TEST(CpuKvOverRoceVerbsConnectionTest, AShutdownConnectionRejectsRequests)
{
    TestConnection fixture = MakeConnection(2);
    std::string error;
    ASSERT_TRUE(fixture.connection->PostReceiveRing(error).ok()) << error;

    fixture.connection->Shutdown();
    std::uint32_t sqe[roce::kSqeHeaderDwords] = {};
    std::uint8_t flag[kResponseCapacity] = {};
    EXPECT_EQ(StatusCode::CONNECTION_ERROR,
              fixture.connection
                  ->PostRequest(sqe, sizeof(sqe), kLkey, flag, kCid, kResponseCapacity, error)
                  .code);
}

}  // namespace
}  // namespace kv::test
