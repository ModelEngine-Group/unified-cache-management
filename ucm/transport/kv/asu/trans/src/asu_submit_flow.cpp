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
#include <cstdint>
#include <utility>
#include "asu_transport_impl.h"
#include "connection_internal.h"
#include "logger.h"

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
    subBatchContext.state = TransportSubBatchState::COMPLETED;
    subBatchContext.status = status;
}

}  // namespace

Status AsuTransportImpl::SubmitTaskRequests(const TransportTaskContext& ctx,
                                            std::vector<TransportSubBatchContext>& subBatchContexts)
{
    Status finalStatus = Status::OK();

    if (IsEntryBatchOp(ctx.opType)) {
        const auto subBatches = ioScheduler_.SplitForAsu(ctx.entries, ctx.opType);
        subBatchContexts.reserve(subBatches.size());
        for (std::size_t index = 0; index < subBatches.size(); ++index) {
            const auto& subBatch = subBatches[index];
            auto& subBatchContext = subBatchContexts.emplace_back();
            auto status = SubmitEntrySubBatchRequest(ctx.opType, subBatch, subBatchContext);
            subBatchContext.status = status;
            if (!status.ok()) {
                UC_ERROR("Submit entry sub-batch failed index={} batch_size={} code={} message={}",
                         index, subBatch.entries.size, static_cast<int>(status.code),
                         status.message);
                if (finalStatus.ok()) { finalStatus = status; }
            }
        }
    } else if (IsKeyBatchOp(ctx.opType)) {
        const auto subBatches = ioScheduler_.SplitForAsu(ctx.keys, ctx.opType);
        subBatchContexts.reserve(subBatches.size());
        for (std::size_t index = 0; index < subBatches.size(); ++index) {
            const auto& subBatch = subBatches[index];
            auto& subBatchContext = subBatchContexts.emplace_back();
            auto status = SubmitKeySubBatchRequest(ctx.opType, subBatch, subBatchContext);
            subBatchContext.status = status;
            if (!status.ok()) {
                UC_ERROR("Submit key sub-batch failed index={} batch_size={} code={} message={}",
                         index, subBatch.keys.size, static_cast<int>(status.code), status.message);
                if (finalStatus.ok()) { finalStatus = status; }
            }
        }
    } else if (IsKeepAliveOp(ctx.opType)) {
        auto& subBatchContext = subBatchContexts.emplace_back();
        auto status = SubmitKeepAliveRequest(subBatchContext);
        subBatchContext.status = status;
        if (!status.ok()) {
            UC_ERROR("Submit keep-alive request failed code={} message={}",
                     static_cast<int>(status.code), status.message);
            if (finalStatus.ok()) { finalStatus = status; }
        }
    } else {
        finalStatus = Status::Error(StatusCode::UNSUPPORTED, "transport operation is unsupported");
        UC_ERROR("Unsupported transport operation op_type={}", static_cast<int>(ctx.opType));
    }
    return finalStatus;
}

Status AsuTransportImpl::BuildSubBatchSendBuffers(
    std::vector<TransportSubBatchContext>& subBatchContexts, std::vector<SendIoBatch>& ioBatches,
    std::vector<std::size_t>& subBatchIndexes)
{
    Status finalStatus = Status::OK();
    ioBatches.reserve(subBatchContexts.size());
    subBatchIndexes.reserve(subBatchContexts.size());

    for (std::size_t index = 0; index < subBatchContexts.size(); ++index) {
        auto& subBatchContext = subBatchContexts[index];
        if (!subBatchContext.status.ok()) {
            UC_ERROR("Skip sub-batch before send index={} cid={} code={} message={}", index,
                     subBatchContext.cid, static_cast<int>(subBatchContext.status.code),
                     subBatchContext.status.message);
            if (finalStatus.ok()) {
                finalStatus = Status::Error(StatusCode::PARTIAL_FAILED,
                                            "one or more sub-batches failed before send");
            }
            ReleaseSubBatchResources(subBatchContext);
            continue;
        }

        if (subBatchContext.flagBuffer.addr == 0 || subBatchContext.flagBuffer.length == 0) {
            const auto status =
                Status::Error(StatusCode::NOT_INITIALIZED, "sub-batch flag buffer is not ready");
            UC_ERROR(
                "Sub-batch flag buffer is not ready index={} cid={} flag_addr={} flag_length={}",
                index, subBatchContext.cid, subBatchContext.flagBuffer.addr,
                subBatchContext.flagBuffer.length);
            SetSubBatchSendFailed(subBatchContext, status);
            if (finalStatus.ok()) { finalStatus = status; }
            ReleaseSubBatchResources(subBatchContext);
            continue;
        }

        ioBatches.push_back(
            SendIoBatch{subBatchContext.channel->GetNativeQp(), &subBatchContext.sendSge});
        subBatchIndexes.emplace_back(index);
    }

    return finalStatus;
}

Status AsuTransportImpl::SendSubBatchBuffers(
    std::vector<TransportSubBatchContext>& subBatchContexts,
    const std::vector<SendIoBatch>& ioBatches, const std::vector<std::size_t>& subBatchIndexes)
{
    Status finalStatus = Status::OK();
    if (ioBatches.empty()) { return finalStatus; }

    const auto kernelCount = GetSendCountAttr(config_.attrs, "kernel_count");
    const auto quietCount = GetSendCountAttr(config_.attrs, "quiet_count");

    const auto sendStatuses = Send(ioBatches, kernelCount, quietCount);
    if (sendStatuses.size() != ioBatches.size()) {
        const auto status = Status::Error(StatusCode::INTERNAL_ERROR,
                                          "transport send returned unexpected status count");
        UC_ERROR("Transport send returned unexpected status count expected={} actual={}",
                 ioBatches.size(), sendStatuses.size());
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

        UC_ERROR("Send sub-batch failed sub_batch_index={} cid={} code={} message={}",
                 subBatchIndexes[index], subBatchContext.cid, static_cast<int>(status.code),
                 status.message);
        SetSubBatchSendFailed(subBatchContext, status);
        connManager_.ReportFailure(subBatchContext.channel);
        if (finalStatus.ok()) { finalStatus = status; }
    }
    return finalStatus;
}

}  // namespace UC::ASU
