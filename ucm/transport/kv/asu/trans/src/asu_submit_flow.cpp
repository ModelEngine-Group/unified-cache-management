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
#include "asu_submit_flow.h"
#include <algorithm>
#include <cstdint>
#include <utility>
#include "connection_internal.h"
#include "transport_task_completion.h"

namespace UC::ASU {

namespace {

std::uint32_t GetSendCountAttr(const std::unordered_map<std::string, std::string>& attrs,
                               const std::string& name)
{
    return static_cast<std::uint32_t>(std::stoull(attrs.at(name), nullptr, 0));
}

void SetSubBatchSendFailed(TransportSubBatchContext& subBatchContext, const Status& status)
{
    std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(), status);
    subBatchContext.state = TransportSubBatchState::FAILED;
    subBatchContext.status = status;
}

}  // namespace

Status SubmitTaskRequests(const TransportTaskContext& ctx, const IoScheduler& ioScheduler,
                          const std::unordered_map<std::string, std::string>& attrs,
                          const SqeCidAllocator& allocateSqeCid, BufferManager& sendBufferManager,
                          BufferManager& flagBufferManager, ProtocolManager& protocolManager,
                          std::vector<TransportSubBatchContext>& subBatchContexts)
{
    Status finalStatus = Status::OK();
    if (IsEntryBatchOp(ctx.opType)) {
        const auto subBatches = ioScheduler.SplitForAsu(ctx.entries, GetSqeBatchLimit(ctx.opType));
        subBatchContexts.reserve(subBatches.size());
        for (const auto& subBatch : subBatches) {
            TransportSubBatchContext subBatchContext;
            auto status = SubmitEntrySubBatchRequest(ctx.opType, subBatch, attrs, allocateSqeCid,
                                                     sendBufferManager, flagBufferManager,
                                                     protocolManager, subBatchContext);
            subBatchContext.status = status;
            if (!status.ok() && finalStatus.ok()) { finalStatus = status; }
            subBatchContexts.push_back(std::move(subBatchContext));
        }
    } else if (IsKeyBatchOp(ctx.opType)) {
        const auto subBatches = ioScheduler.SplitForAsu(ctx.keys, GetSqeBatchLimit(ctx.opType));
        subBatchContexts.reserve(subBatches.size());
        for (const auto& subBatch : subBatches) {
            TransportSubBatchContext subBatchContext;
            auto status = SubmitKeySubBatchRequest(ctx.opType, subBatch, attrs, allocateSqeCid,
                                                   sendBufferManager, flagBufferManager,
                                                   protocolManager, subBatchContext);
            subBatchContext.status = status;
            if (!status.ok() && finalStatus.ok()) { finalStatus = status; }
            subBatchContexts.push_back(std::move(subBatchContext));
        }
    } else if (ctx.opType == TransportOpType::KEEP_ALIVE) {
        TransportSubBatchContext subBatchContext;
        auto status = SubmitKeepAliveRequest(allocateSqeCid, sendBufferManager, flagBufferManager,
                                             protocolManager, subBatchContext);
        subBatchContext.status = status;
        if (!status.ok() && finalStatus.ok()) { finalStatus = status; }
        subBatchContexts.push_back(std::move(subBatchContext));
    } else {
        finalStatus = Status::Error(StatusCode::UNSUPPORTED, "transport operation is unsupported");
    }
    return finalStatus;
}

Status BuildSubBatchSendBuffers(std::vector<TransportSubBatchContext>& subBatchContexts,
                                std::vector<SendIoBatch>& ioBatches,
                                std::vector<std::size_t>& subBatchIndexes,
                                BufferManager& sendBufferManager, BufferManager& flagBufferManager)
{
    Status finalStatus = Status::OK();
    ioBatches.reserve(subBatchContexts.size());
    subBatchIndexes.reserve(subBatchContexts.size());

    for (std::size_t index = 0; index < subBatchContexts.size(); ++index) {
        auto& subBatchContext = subBatchContexts[index];
        if (subBatchContext.state == TransportSubBatchState::FAILED) {
            if (finalStatus.ok()) {
                finalStatus = Status::Error(StatusCode::PARTIAL_FAILED,
                                            "one or more sub-batches failed before send");
            }
            const auto releaseStatus =
                ReleaseSubBatchResources(subBatchContext, sendBufferManager, flagBufferManager);
            if (finalStatus.ok() && !releaseStatus.ok()) { finalStatus = releaseStatus; }
            continue;
        }

        if (subBatchContext.flagBuffer.addr == 0 || subBatchContext.flagBuffer.length == 0) {
            const auto status =
                Status::Error(StatusCode::NOT_INITIALIZED, "sub-batch flag buffer is not ready");
            SetSubBatchSendFailed(subBatchContext, status);
            if (finalStatus.ok()) { finalStatus = status; }
            const auto releaseStatus =
                ReleaseSubBatchResources(subBatchContext, sendBufferManager, flagBufferManager);
            if (finalStatus.ok() && !releaseStatus.ok()) { finalStatus = releaseStatus; }
            continue;
        }

        ioBatches.push_back(
            SendIoBatch{subBatchContext.channel->GetNativeQp(), &subBatchContext.sendSge});
        subBatchIndexes.push_back(index);
    }

    return finalStatus;
}

Status SendSubBatchBuffers(std::vector<TransportSubBatchContext>& subBatchContexts,
                           const std::vector<SendIoBatch>& ioBatches,
                           const std::vector<std::size_t>& subBatchIndexes,
                           const std::unordered_map<std::string, std::string>& attrs,
                           ConnectionManager& connManager)
{
    Status finalStatus = Status::OK();
    if (ioBatches.empty()) { return finalStatus; }

    const auto kernelCount = GetSendCountAttr(attrs, "kernel_count");
    const auto quietCount = GetSendCountAttr(attrs, "quiet_count");

    const auto sendStatuses = Send(ioBatches, kernelCount, quietCount);
    if (sendStatuses.size() != ioBatches.size()) {
        const auto status = Status::Error(StatusCode::INTERNAL_ERROR,
                                          "transport send returned unexpected status count");
        for (auto index : subBatchIndexes) {
            auto& subBatchContext = subBatchContexts[index];
            SetSubBatchSendFailed(subBatchContext, status);
        }
        return status;
    }

    for (std::size_t index = 0; index < sendStatuses.size(); ++index) {
        auto& subBatchContext = subBatchContexts[subBatchIndexes[index]];
        const auto& status = sendStatuses[index];
        if (status.ok()) { continue; }

        SetSubBatchSendFailed(subBatchContext, status);
        connManager.ReportFailure(subBatchContext.channel);
        if (finalStatus.ok()) { finalStatus = status; }
    }
    return finalStatus;
}

}  // namespace UC::ASU
