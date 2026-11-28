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
#include "verbs_connection.h"
#include <algorithm>
#include <cstring>
#include <utility>
#include "logger.h"
#include "roce_io_wire.h"

namespace kv {
namespace roce {
namespace {

constexpr std::size_t kPollBatch = 16;

// Work request ids of one queue pair: the low bit tells a send completion from
// a receive one, so both share the single completion queue of the pair.
constexpr std::uint64_t kSendWrBit = 1U;

std::uint64_t ReceiveWrId(std::uint32_t index)
{
    return static_cast<std::uint64_t>(index) << 1U;
}

std::uint64_t SendWrId(std::uint64_t tag) { return (tag << 1U) | kSendWrBit; }

std::uint32_t SlotFromReceiveWrId(std::uint64_t wrId)
{
    return static_cast<std::uint32_t>(wrId >> 1U);
}

StatusCode StatusCodeOf(const Status& status, StatusCode fallback)
{
    return status.code == StatusCode::OK ? fallback : status.code;
}

std::string HexDump(const void* data, std::size_t length)
{
    static constexpr char kHexDigits[] = "0123456789abcdef";
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::string text;
    text.reserve(length * 3U);
    for (std::size_t index = 0U; index < length; ++index) {
        if (index != 0U) { text.push_back(' '); }
        text.push_back(kHexDigits[bytes[index] >> 4U]);
        text.push_back(kHexDigits[bytes[index] & 0x0FU]);
    }
    return text;
}

}  // namespace

Status VerbQueuePairIo::PostSend(const void* buffer, std::size_t length, std::uint32_t lkey,
                                 std::uint64_t wrId, std::string& error)
{
    if (queuePair_ == nullptr) {
        error = "queue pair is not set";
        return Status::Error(StatusCode::NOT_INITIALIZED, error);
    }
    return queuePair_->PostSend(wrId, buffer, length, lkey, error);
}

Status VerbQueuePairIo::PostReceive(void* buffer, std::size_t length, std::uint32_t lkey,
                                    std::uint64_t wrId, std::string& error)
{
    if (queuePair_ == nullptr) {
        error = "queue pair is not set";
        return Status::Error(StatusCode::NOT_INITIALIZED, error);
    }
    return queuePair_->PostReceive(wrId, buffer, length, lkey, error);
}

std::size_t VerbQueuePairIo::Poll(Completion* completions, std::size_t maxCompletions)
{
    if (queuePair_ == nullptr) { return 0U; }
    return queuePair_->Poll(completions, maxCompletions);
}

VerbsConnection::~VerbsConnection() { Shutdown(); }

Status VerbsConnection::Open(VerbDevice& device, VerbQp& queuePair, std::uint32_t receiveDepth,
                             std::unique_ptr<VerbsConnection>& connection, std::string& error)
{
    error.clear();
    if (receiveDepth == 0U) {
        error = "receive depth must not be zero";
        return Status::Error(StatusCode::INVALID_ARGUMENT, error);
    }

    auto opened = std::unique_ptr<VerbsConnection>(new VerbsConnection());
    opened->device_ = &device;
    opened->receiveDepth_ = receiveDepth;
    opened->receiveStorage_.assign(static_cast<std::size_t>(receiveDepth) * kResponseRecordBytes,
                                   0U);

    std::uint32_t rkey = 0;
    Status registered =
        device.RegisterMemory(opened->receiveStorage_.data(), opened->receiveStorage_.size(),
                              kMemoryAccessLocalWrite, opened->receiveRegion_, rkey, error);
    if (!registered.ok()) {
        opened->receiveStorage_.clear();
        return registered;
    }
    opened->receiveLkey_ = device.Lkey(opened->receiveRegion_);
    if (opened->receiveLkey_ == 0U) {
        error = "the receive ring has no local key";
        opened->Shutdown();
        return Status::Error(StatusCode::IO_ERROR, error);
    }

    opened->queuePairIo_ = std::make_unique<VerbQueuePairIo>(queuePair);
    opened->receiveSlots_.assign(receiveDepth, ReceiveSlot{});
    for (std::uint32_t index = 0; index < receiveDepth; ++index) {
        opened->receiveSlots_[index].buffer =
            opened->receiveStorage_.data() +
            static_cast<std::size_t>(index) * kResponseRecordBytes;
    }

    Status posted = opened->PostReceiveRing(error);
    if (!posted.ok()) {
        opened->Shutdown();
        return posted;
    }
    connection = std::move(opened);
    return Status::OK();
}

std::unique_ptr<VerbsConnection> VerbsConnection::OpenForTest(
    std::unique_ptr<QueuePairIo> queuePairIo, std::uint32_t receiveDepth, std::uint32_t lkey)
{
    auto connection = std::unique_ptr<VerbsConnection>(new VerbsConnection());
    connection->queuePairIo_ = std::move(queuePairIo);
    connection->receiveDepth_ = receiveDepth;
    connection->receiveLkey_ = lkey;
    connection->receiveStorage_.assign(static_cast<std::size_t>(receiveDepth) * kResponseRecordBytes,
                                       0U);
    connection->receiveSlots_.assign(receiveDepth, ReceiveSlot{});
    for (std::uint32_t index = 0; index < receiveDepth; ++index) {
        connection->receiveSlots_[index].buffer =
            connection->receiveStorage_.data() +
            static_cast<std::size_t>(index) * kResponseRecordBytes;
    }
    return connection;
}

void VerbsConnection::Shutdown() noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    failed_.store(true, std::memory_order_release);
    inflight_.clear();
    if (receiveRegion_ != nullptr && device_ != nullptr) {
        std::string ignored;
        static_cast<void>(device_->UnregisterMemory(receiveRegion_, ignored));
    }
    receiveRegion_ = nullptr;
    receiveLkey_ = 0;
    receiveSlots_.clear();
    receiveStorage_.clear();
    queuePairIo_.reset();
    device_ = nullptr;
}

Status VerbsConnection::PostReceiveRing(std::string& error)
{
    error.clear();
    std::lock_guard<std::mutex> guard(mutex_);
    for (std::uint32_t index = 0; index < receiveSlots_.size(); ++index) {
        const Status posted = PostReceiveSlot(index);
        if (!posted.ok()) {
            error = posted.message;
            return posted;
        }
    }
    return Status::OK();
}

Status VerbsConnection::PostRequest(const void* sqe, std::size_t length, std::uint32_t lkey,
                                    void* flagBuffer, std::uint16_t cid,
                                    std::size_t responseBytes, std::string& error)
{
    error.clear();
    if (sqe == nullptr || length == 0U) {
        error = "send buffer is empty";
        return Status::Error(StatusCode::INVALID_ARGUMENT, error);
    }
    if (flagBuffer == nullptr) {
        error = "flag buffer is null";
        return Status::Error(StatusCode::INVALID_ARGUMENT, error);
    }
    if (cid == 0U) {
        error = "request cid is zero";
        return Status::Error(StatusCode::INVALID_ARGUMENT, error);
    }
    if (responseBytes < kResponseHeaderBytes) {
        error = "response slot is too small for a completion header";
        return Status::Error(StatusCode::INVALID_ARGUMENT, error);
    }

    std::lock_guard<std::mutex> guard(mutex_);
    if (failed_.load(std::memory_order_acquire) || queuePairIo_ == nullptr) {
        error = "connection failed";
        return Status::Error(StatusCode::CONNECTION_ERROR, error);
    }
    if (inflight_.size() >= receiveDepth_) {
        // Every receive work request is claimed by an outstanding answer; the
        // caller retries instead of blocking this thread or growing the ring.
        error = "receive queue is full";
        return Status::Error(StatusCode::RESOURCE_BUSY, error);
    }
    const auto inserted = inflight_.emplace(cid, InFlight{flagBuffer, responseBytes});
    if (!inserted.second) {
        error = "cid " + std::to_string(cid) + " is already in flight";
        return Status::Error(StatusCode::RESOURCE_BUSY, error);
    }

    if (kv::logger::isEnabledFor(kv::logger::Level::DEBUG)) {
        KV_DEBUG("VerbsConnection::PostRequest request cid={} sqe_bytes={} response_bytes={} "
                 "sqe=[{}]",
                 cid, length, responseBytes, HexDump(sqe, length));
    }
    const Status posted =
        queuePairIo_->PostSend(sqe, length, lkey, SendWrId(nextSendTag_++), error);
    if (!posted.ok()) {
        inflight_.erase(inserted.first);
        return Status::Error(StatusCode::IO_ERROR,
                             posted.message.empty() ? "posting the send failed" : posted.message);
    }
    return Status::OK();
}

void VerbsConnection::PollCompletion()
{
    if (failed_.load(std::memory_order_acquire)) { return; }

    std::lock_guard<std::mutex> guard(mutex_);
    if (failed_.load(std::memory_order_acquire) || queuePairIo_ == nullptr) { return; }

    Completion completions[kPollBatch];
    for (;;) {
        const std::size_t count = queuePairIo_->Poll(completions, kPollBatch);
        if (count == 0U) { return; }
        for (std::size_t index = 0; index < count; ++index) {
            const Completion& completion = completions[index];
            if (completion.type == CompletionType::Send) {
                if (completion.result != CompletionResult::Success) {
                    Fail();
                    return;
                }
                continue;
            }

            const std::uint32_t slot = SlotFromReceiveWrId(completion.wrId);
            if (slot >= receiveSlots_.size()) {
                Fail();
                return;
            }
            receiveSlots_[slot].posted = false;
            if (completion.result != CompletionResult::Success) {
                Fail();
                return;
            }

            const auto* response = static_cast<const std::uint32_t*>(receiveSlots_[slot].buffer);
            const std::uint16_t cid = static_cast<std::uint16_t>(
                __atomic_load_n(response + 3, __ATOMIC_ACQUIRE) & kSqeCidMask);
            const auto found = inflight_.find(cid);
            if (found != inflight_.end()) {
                // The answer is copied before the slot is reposted: the repost
                // recycles the buffer the copy still reads from.
                const std::size_t received =
                    std::min<std::size_t>(completion.byteLength, kResponseRecordBytes);
                if (kv::logger::isEnabledFor(kv::logger::Level::DEBUG)) {
                    KV_DEBUG(
                        "VerbsConnection::PollCompletion answer cid={} byte_length={} received={} "
                        "capacity={} record=[{}]",
                        cid, completion.byteLength, received, found->second.responseBytes,
                        HexDump(response, received));
                }
                const Status published = PublishResponse(response, received,
                                                         found->second.flagBuffer,
                                                         found->second.responseBytes);
                if (!published.ok()) {
                    Fail();
                    return;
                }
                inflight_.erase(found);
            }

            const Status reposted = PostReceiveSlot(slot);
            if (!reposted.ok()) {
                Fail();
                return;
            }
        }
    }
}

bool VerbsConnection::Failed() const noexcept { return failed_.load(std::memory_order_acquire); }

std::uint32_t VerbsConnection::FreeReceiveSlots() const noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (inflight_.size() >= receiveDepth_) { return 0U; }
    return receiveDepth_ - static_cast<std::uint32_t>(inflight_.size());
}

std::uint32_t VerbsConnection::ReceiveDepth() const noexcept { return receiveDepth_; }

Status VerbsConnection::PostReceiveSlot(std::uint32_t index)
{
    if (queuePairIo_ == nullptr || index >= receiveSlots_.size()) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "receive slot does not exist");
    }
    ReceiveSlot& slot = receiveSlots_[index];
    // Zero the buffer before it is posted again, so a short answer can never
    // expose bytes of the previous one.
    std::memset(slot.buffer, 0, kResponseRecordBytes);
    std::string error;
    const Status posted =
        queuePairIo_->PostReceive(slot.buffer, kResponseRecordBytes, receiveLkey_,
                                  ReceiveWrId(index), error);
    if (!posted.ok()) {
        return Status::Error(StatusCodeOf(posted, StatusCode::IO_ERROR),
                             "posting a receive buffer failed: " + error);
    }
    slot.posted = true;
    return Status::OK();
}

void VerbsConnection::Fail() noexcept { failed_.store(true, std::memory_order_release); }

}  // namespace roce
}  // namespace kv
