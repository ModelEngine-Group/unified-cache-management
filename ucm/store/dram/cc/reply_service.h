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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_REPLY_SERVICE_H
#define UNIFIEDCACHE_DRAM_STORE_CC_REPLY_SERVICE_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include "kv_protocol.h"
#include "messages.h"
#include "pool/buffer_pool.h"
#include "status/status.h"

namespace UC::Dram {

struct ReplyMemoryRegion final {
    void* address{nullptr};
    std::size_t length{0};
};

// ReplyService is the single owner of reply-slot allocation and observation.
// A slot is reusable only after NodeActor releases it at a safe terminal point
// (reply observed, definitely-not-transmitted, or epoch fenced). That safety rule makes
// a separate slot generation unnecessary.
class ReplyService final {
public:
    struct Options {
        std::uint32_t slotSize{0};
        std::size_t slotCount{0};
        std::chrono::microseconds pollInterval{0};
        NodeEventPublisher publishEvent;
    };

    static Expected<std::unique_ptr<ReplyService>> Create(Options options);
    ~ReplyService();

    ReplyService(const ReplyService&) = delete;
    ReplyService& operator=(const ReplyService&) = delete;

    Status Start();
    Status Shutdown();

    Expected<ReplySlot> Acquire(const RequestToken& token, OpType op, std::size_t entryCount);
    Status Release(const RequestToken& token, const ReplySlot& slot) noexcept;

    ReplyMemoryRegion MemoryRegion() const noexcept;
    std::size_t Available() const;

private:
    static constexpr std::size_t kNoIndex = std::numeric_limits<std::size_t>::max();

    struct Lease {
        RequestToken token;
        ReplySlot slot;
        OpType op{OpType::LOOKUP};
        std::size_t entryCount{0};
        std::size_t payloadSize{0};
        bool delivered{false};
    };

    struct SlotContext {
        std::mutex mutex;
        std::optional<Lease> lease;
    };

    explicit ReplyService(Options options);
    Status Init();
    static DramPool::KvOpcode ToOpcode(OpType op) noexcept;
    std::size_t ReplyPayloadSize(OpType op, std::size_t entryCount) const noexcept;
    bool CompletionReady(const Lease& lease) const noexcept;
    Status DecodeReply(const Lease& lease, std::vector<EntryResult>* entryResults);
    void Run() noexcept;

    Options options_;
    DramPool::ProtocolManager protocol_;
    BufferPool buffers_;
    mutable std::mutex activeLeasesMutex_;
    std::condition_variable wake_;
    std::unique_ptr<SlotContext[]> slotContexts_;
    std::vector<std::size_t> activeLeaseIndices_;
    std::vector<std::size_t> activeLeasePositions_;
    std::uint64_t activeLeaseVersion_{0};
    std::thread worker_;
    // False rejects new leases and asks the polling worker to stop.
    std::atomic<bool> acceptingLeases_{false};
};

}  // namespace UC::Dram

#endif  // UNIFIEDCACHE_DRAM_STORE_CC_REPLY_SERVICE_H
