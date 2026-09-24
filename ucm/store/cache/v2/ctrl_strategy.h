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

#include <chrono>
#include <cstddef>
#include <string>
#include <thread>
#include "ctrl_layout.h"
#include "global_config.h"
#include "ipc/fd_socket.h"
#include "ipc/mem_fd.h"
#include "status/status.h"

namespace UC::Cache2 {

class CtrlStrategy {
    struct Dimensions {
        size_t rankCount{0};
        size_t slotsPerRank{0};
        size_t slotSize{0};
        size_t bucketCount{0};
    };

    MemFd ctrlMem_;
    FdSocket socket_;
    std::thread acceptThread_;
    CtrlLayout layout_;
    std::string socketName_;

public:
    ~CtrlStrategy()
    {
        socket_.Close();
        if (acceptThread_.joinable()) { acceptThread_.join(); }
    }

    CtrlStrategy() = default;
    CtrlStrategy(const CtrlStrategy&) = delete;
    CtrlStrategy& operator=(const CtrlStrategy&) = delete;

    // Requires localRankSize in [1, kMaxRanks], checked by Buffer::Setup.
    Status Setup(const Config& cfg)
    {
        if (cfg.uniqueId.empty()) { return Status::InvalidParam("cache2 uniqueId is empty"); }
        socketName_ = "ucm_cache2_" + cfg.uniqueId + "_ctrl";
        auto s = socket_.Listen(socketName_);
        if (s.Success()) {
            auto result = SetupCreator(cfg);
            if (result.Failure()) { socket_.Close(); }
            return result;
        }
        if (s != Status::DuplicateKey()) { return s; }
        return SetupJoiner(cfg);
    }

    CtrlLayout& Layout() { return layout_; }
    const CtrlLayout& Layout() const { return layout_; }

private:
    static Status ResolveDimensions(const Config& cfg, Dimensions& dims)
    {
        /* A5 has one data partition per local worker. The public configuration
         * describes node-wide capacity; the control layout divides it evenly. */
        if (cfg.shardSize == 0) { return Status::InvalidParam("invalid cache2 shard size(0)"); }
        auto slotCount = cfg.bufferCapacity / cfg.shardSize;
        if (slotCount < cfg.localRankSize) {
            return Status::InvalidParam(
                "cache2 buffer capacity({}) is too small for shard size({}) and ranks({})",
                cfg.bufferCapacity, cfg.shardSize, cfg.localRankSize);
        }
        dims.rankCount = cfg.localRankSize;
        dims.slotsPerRank = slotCount / dims.rankCount;
        dims.slotSize = cfg.shardSize;
        slotCount = dims.rankCount * dims.slotsPerRank;
        dims.bucketCount = CtrlLayout::RecommendBucketCount(slotCount);
        return Status::OK();
    }

    Status SetupCreator(const Config& cfg)
    {
        Dimensions dims;
        auto s = ResolveDimensions(cfg, dims);
        if (s.Failure()) { return s; }
        auto slotCount = dims.rankCount * dims.slotsPerRank;
        auto lockCount = CtrlLayout::LockStripeCount(dims.bucketCount);
        auto totalSize = CtrlLayout::TotalSize(dims.bucketCount, lockCount, slotCount);
        s = ctrlMem_.Create("ucm_cache2_ctrl", totalSize);
        if (s.Failure()) { return s; }
        layout_.Bind(ctrlMem_.Addr(), dims.rankCount, dims.slotsPerRank, dims.bucketCount,
                     lockCount);
        layout_.InitHeader(dims.slotSize);
        layout_.MarkReady();
        acceptThread_ =
            std::thread([this, joinerCount = dims.rankCount] { AcceptLoop(joinerCount); });
        return Status::OK();
    }

    Status SetupJoiner(const Config& cfg)
    {
        Dimensions expected;
        const bool discoverSlotSize = cfg.deviceId < 0 && cfg.shardSize == 0;
        if (!discoverSlotSize) {
            auto s = ResolveDimensions(cfg, expected);
            if (s.Failure()) { return s; }
        }
        constexpr auto backoff = std::chrono::milliseconds(10);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(cfg.timeoutMs);
        for (;;) {
            auto s = socket_.Connect(socketName_);
            if (s.Success()) { break; }
            if (cfg.timeoutMs > 0 && std::chrono::steady_clock::now() >= deadline) {
                return Status::Retry();
            }
            std::this_thread::sleep_for(backoff);
        }

        int32_t fd = -1;
        auto s = socket_.RecvFd(fd);
        socket_.Close();
        if (s.Failure()) { return s; }
        s = ctrlMem_.Adopt(fd, sizeof(CtrlLayout::Header));
        if (s.Failure()) { return s; }
        layout_.Bind(ctrlMem_.Addr(), 0, 0, 0, 0);
        if (!layout_.WaitReady(cfg.timeoutMs)) { return Status::Retry(); }

        auto* header = layout_.Hdr();
        if (discoverSlotSize) {
            // Keep the scheduler's capacity and rank checks; only the shard size is unknown.
            auto resolved = cfg;
            resolved.shardSize = header->slotSize;
            s = ResolveDimensions(resolved, expected);
            if (s.Failure()) { return s; }
        }
        auto rankCount = header->rankCount;
        auto slotsPerRank = header->slotsPerRank;
        auto bucketCount = header->bucketCount;
        auto lockCount = header->lockStripeCount;
        if (expected.rankCount != rankCount || expected.slotsPerRank != slotsPerRank ||
            expected.slotSize != header->slotSize || expected.bucketCount != bucketCount ||
            lockCount != CtrlLayout::LockStripeCount(expected.bucketCount)) {
            return Status::InvalidParam("cache2 participants disagree on control layout");
        }

        auto slotCount = rankCount * slotsPerRank;
        s = ctrlMem_.Remap(CtrlLayout::TotalSize(bucketCount, lockCount, slotCount));
        if (s.Failure()) { return s; }
        layout_.Bind(ctrlMem_.Addr(), rankCount, slotsPerRank, bucketCount, lockCount);
        return Status::OK();
    }

    void AcceptLoop(size_t joinerCount)
    {
        for (size_t joined = 0; joined < joinerCount; ++joined) {
            if (socket_.AcceptAndSend(ctrlMem_.Fd()).Failure()) { break; }
        }
        socket_.Close();
    }
};

}  // namespace UC::Cache2
