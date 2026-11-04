/**
 * MIT License
 *
 * Copyright (c) 2026 relat-ivity
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
#ifndef COPY_INSTANCE_GDR_H
#define COPY_INSTANCE_GDR_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "copy_buffer_gdr.h"
#include "copy_instance.h"
#include "error_handle_gdr.h"
#include "rdma_channel_gdr.h"
#include "submit_executor_gdr.h"

class GdrCopyInstance : public CopyInstance {
    struct PendingWrite {
        int32_t deviceId = -1;
        uint64_t lastWorkRequestId = 0;
    };

    std::vector<const GdrCopyBuffer*> srcBuffers_;
    std::vector<const GdrCopyBuffer*> dstBuffers_;

    void Prepare(const std::vector<const CopyBuffer*>& srcBuffers,
                 const std::vector<const CopyBuffer*>& dstBuffers) override
    {
        ASSERT(srcBuffers.size() == dstBuffers.size());
        srcBuffers_.clear();
        dstBuffers_.clear();
        srcBuffers_.reserve(srcBuffers.size());
        dstBuffers_.reserve(dstBuffers.size());
        for (size_t i = 0; i < srcBuffers.size(); ++i) {
            const auto* src = dynamic_cast<const GdrCopyBuffer*>(srcBuffers[i]);
            const auto* dst = dynamic_cast<const GdrCopyBuffer*>(dstBuffers[i]);
            ASSERT(src != nullptr);
            ASSERT(dst != nullptr);
            ASSERT(src->Size() == dst->Size());
            ASSERT(src->Number() == dst->Number());
            srcBuffers_.push_back(src);
            dstBuffers_.push_back(dst);
        }
    }

    std::pair<size_t, size_t> DoCopyOnce() override
    {
        using namespace std::chrono;

        std::vector<PendingWrite> pendingWrites(srcBuffers_.size());
        const auto copyStart = steady_clock::now();
        const auto submitStart = copyStart;
        if (srcBuffers_.size() <= 1) {
            for (size_t i = 0; i < srcBuffers_.size(); ++i) {
                pendingWrites[i] = SubmitWrite(*srcBuffers_[i], *dstBuffers_[i]);
            }
        } else {
            SubmitExecutor::Instance().Run(srcBuffers_.size(), [&](size_t index) {
                pendingWrites[index] = SubmitWrite(*srcBuffers_[index], *dstBuffers_[index]);
            });
        }
        const auto submitCost =
            duration_cast<microseconds>(steady_clock::now() - submitStart).count();

        std::unordered_map<int32_t, uint64_t> waitTargets;
        for (const auto& pending : pendingWrites) {
            if (pending.lastWorkRequestId == 0) { continue; }
            auto& target = waitTargets[pending.deviceId];
            target = std::max(target, pending.lastWorkRequestId);
        }
        for (const auto& waitTarget : waitTargets) {
            ChannelManager::Instance().Get(waitTarget.first).Wait(waitTarget.second);
        }
        const auto copyCost = duration_cast<microseconds>(steady_clock::now() - copyStart).count();
        return {copyCost, submitCost};
    }

    void Cleanup() override
    {
        srcBuffers_.clear();
        dstBuffers_.clear();
    }

    static PendingWrite SubmitWrite(const GdrCopyBuffer& src, const GdrCopyBuffer& dst)
    {
        const auto targetDevice = dst.Device();
        ASSERT(src.HasMR(targetDevice));
        ASSERT(dst.HasMR(targetDevice));

        auto& channel = ChannelManager::Instance().Get(static_cast<int32_t>(targetDevice));
        PendingWrite pending;
        pending.deviceId = static_cast<int32_t>(targetDevice);
        for (size_t i = 0; i < src.Number(); ++i) {
            pending.lastWorkRequestId =
                channel.SubmitWrite(src.Address(i), src.LKey(targetDevice), dst.Address(i),
                                    dst.RKey(targetDevice), src.Size());
        }
        return pending;
    }

public:
    explicit GdrCopyInstance(size_t iterations) : CopyInstance(iterations, false) {}
    std::string Name() const override { return "GDR"; }
};

#endif  // COPY_INSTANCE_GDR_H
