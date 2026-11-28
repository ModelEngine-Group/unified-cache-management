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
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "kv_types.h"
#include "roce_verb_qp.h"

namespace kv {
namespace roce {

class VerbDevice;
class VerbQp;

// Largest answer the store appliance sends for one request: the sixteen byte
// completion header plus its result buffer (the recv_data_block layout).
constexpr std::size_t kResponseRecordBytes = 80;

// The two sided data path of one queue pair.
//
// Send side: PostRequest() posts the packed SQE with an RDMA send.  It never
// waits for, polls or sleeps on the completion queue; when the receive queue has
// no free slot left it reports RESOURCE_BUSY and the caller retries.  The request
// is registered under its cid before the send is posted, so an answer can never
// outrun the bookkeeping.
//
// Receive side: the whole receive ring is posted when the connection is opened.
// PollCompletion() reaps the completion queue, matches each answer to its request
// by cid (answers are not ordered like the requests), copies the completion
// record into the flag buffer the caller polls and only then reposts the slot.
//
// The transport calls PollCompletion() from its completion loop; the provider
// creates no thread of its own.
class VerbsConnection {
public:
    // Verbs primitives of one queue pair.  They are virtual so a unit test can
    // drive the bookkeeping with a fake instead of a device.
    class QueuePairIo {
    public:
        virtual ~QueuePairIo() = default;

        virtual Status PostSend(const void* buffer, std::size_t length, std::uint32_t lkey,
                                std::uint64_t wrId, std::string& error) = 0;

        virtual Status PostReceive(void* buffer, std::size_t length, std::uint32_t lkey,
                                   std::uint64_t wrId, std::string& error) = 0;

        virtual std::size_t Poll(Completion* completions, std::size_t maxCompletions) = 0;
    };

    VerbsConnection() = default;

    ~VerbsConnection();

    VerbsConnection(const VerbsConnection&) = delete;
    VerbsConnection& operator=(const VerbsConnection&) = delete;

    // Allocates and registers the receive ring of an established queue pair and
    // posts it.  The queue pair has to be in RTS already.
    static Status Open(VerbDevice& device, VerbQp& queuePair, std::uint32_t receiveDepth,
                       std::unique_ptr<VerbsConnection>& connection, std::string& error);

    // Same bookkeeping with caller supplied primitives: used by the unit tests.
    static std::unique_ptr<VerbsConnection> OpenForTest(std::unique_ptr<QueuePairIo> queuePairIo,
                                                        std::uint32_t receiveDepth,
                                                        std::uint32_t lkey);

    // Stops using the queue pair and releases the receive memory.  Safe to call
    // twice; the queue pair itself is owned by the caller.
    void Shutdown() noexcept;

    // Posts one request.  `cid` identifies the answer, `flagBuffer` receives the
    // completion record once it arrives and `responseBytes` is how much room it
    // has.  Returns RESOURCE_BUSY when the receive queue is exhausted, IO_ERROR
    // when the queue pair refused the send and INVALID_ARGUMENT for an unusable
    // argument.
    Status PostRequest(const void* sqe, std::size_t length, std::uint32_t lkey, void* flagBuffer,
                       std::uint16_t cid, std::size_t responseBytes, std::string& error);

    // Reaps the completion queue and publishes finished requests.
    void PollCompletion();

    // True once a completion failed or a receive slot could not be reposted; the
    // connection then reports CONNECTION_ERROR to its caller and the transport
    // rebuilds the channel.
    bool Failed() const noexcept;

    // Receive slots not claimed by an in flight request.
    std::uint32_t FreeReceiveSlots() const noexcept;

    std::uint32_t ReceiveDepth() const noexcept;

    // Posts the receive ring.  Called by Open(); exposed for the fake driven tests.
    Status PostReceiveRing(std::string& error);

private:
    struct InFlight {
        void* flagBuffer{nullptr};
        // Bytes the caller's flag buffer holds for this request.
        std::size_t responseBytes{0};
    };

    struct ReceiveSlot {
        void* buffer{nullptr};
        bool posted{false};
    };

    Status PostReceiveSlot(std::uint32_t index);

    void Fail() noexcept;

    mutable std::mutex mutex_;

    std::unique_ptr<QueuePairIo> queuePairIo_;

    VerbDevice* device_{nullptr};

    void* receiveRegion_{nullptr};

    std::uint32_t receiveLkey_{0};

    std::uint32_t receiveDepth_{0};

    std::vector<std::uint8_t> receiveStorage_;

    std::vector<ReceiveSlot> receiveSlots_;

    std::unordered_map<std::uint16_t, InFlight> inflight_;

    std::uint64_t nextSendTag_{1};

    std::atomic<bool> failed_{false};
};

// Queue pair primitives of the real verbs backend.
class VerbQueuePairIo final : public VerbsConnection::QueuePairIo {
public:
    explicit VerbQueuePairIo(VerbQp& queuePair) : queuePair_(&queuePair) {}

    Status PostSend(const void* buffer, std::size_t length, std::uint32_t lkey, std::uint64_t wrId,
                    std::string& error) override;

    Status PostReceive(void* buffer, std::size_t length, std::uint32_t lkey, std::uint64_t wrId,
                       std::string& error) override;

    std::size_t Poll(Completion* completions, std::size_t maxCompletions) override;

private:
    VerbQp* queuePair_;
};

}  // namespace roce
}  // namespace kv
