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
#include "conn/connection_internal.h"
#include "logger.h"
#include "trans_task_executor.h"

namespace UC::ASU {

namespace {

std::uint32_t GetSendCountAttr(const std::unordered_map<std::string, std::string>& attrs,
                               const std::string& name)
{
    auto it = attrs.find(name);
    if (it == attrs.end()) { return 0; }
    return static_cast<std::uint32_t>(std::stoull(it->second, nullptr, 0));
}

void SetSubBatchSendFailed(TransportSubBatchContext& subBatchContext, const Status& status)
{
    std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(), status);
    subBatchContext.state = TransportSubBatchState::COMPLETED;
    subBatchContext.status = status;
}

}  // namespace

Status TransportTaskExecutor::PrepareTaskSubBatches(
    const TransportTask& ctx, std::vector<TransportSubBatchContext>& subBatchContexts)
{
    if (IsEntryBatchOp(ctx.opType)) {
        const auto opType = ctx.opType;
        if (ctx.entries.empty()) {
            UC_ERROR("Submit entry batch failed: entry batch is empty");
            return Status::Error(StatusCode::INVALID_ARGUMENT, "entry batch is empty");
        }
        const auto entries = BatchView<KVBuffer>{ctx.entries.data(), ctx.entries.size()};
        const auto subBatches = ioScheduler_.SplitForAsu(entries, opType);
        subBatchContexts.reserve(subBatches.size());
        for (std::size_t index = 0; index < subBatches.size(); ++index) {
            const auto& subBatch = subBatches[index];
            auto& subBatchContext = subBatchContexts.emplace_back();
            const auto subBatchStatus =
                BuildEntrySubBatchRequest(opType, subBatch, subBatchContext);
            if (!subBatchStatus.ok()) {
                UC_ERROR(
                    "Build entry sub-batch request failed index={} batch_size={} code={} "
                    "message={}",
                    index, subBatch.entries.size, static_cast<int>(subBatchStatus.code),
                    subBatchStatus.message);
                return subBatchStatus;
            }
        }
    } else if (IsKeyBatchOp(ctx.opType)) {
        if (ctx.keys.empty()) {
            UC_ERROR("Submit key batch failed: key batch is empty");
            return Status::Error(StatusCode::INVALID_ARGUMENT, "key batch is empty");
        }
        const auto keys = BatchView<CacheKey>{ctx.keys.data(), ctx.keys.size()};
        const auto subBatches = ioScheduler_.SplitForAsu(keys, ctx.opType);
        subBatchContexts.reserve(subBatches.size());
        for (std::size_t index = 0; index < subBatches.size(); ++index) {
            const auto& subBatch = subBatches[index];
            auto& subBatchContext = subBatchContexts.emplace_back();
            const auto subBatchStatus =
                SubmitKeySubBatchRequest(ctx.opType, subBatch, subBatchContext);
            if (!subBatchStatus.ok()) {
                UC_ERROR("Submit key sub-batch failed index={} batch_size={} code={} message={}",
                         index, subBatch.keys.size, static_cast<int>(subBatchStatus.code),
                         subBatchStatus.message);
                return subBatchStatus;
            }
        }
    } else if (IsKeepAliveOp(ctx.opType)) {
        auto& subBatchContext = subBatchContexts.emplace_back();
        const auto subBatchStatus = SubmitKeepAliveRequest(subBatchContext);
        if (!subBatchStatus.ok()) {
            UC_ERROR("Submit keep-alive request failed code={} message={}",
                     static_cast<int>(subBatchStatus.code), subBatchStatus.message);
            return subBatchStatus;
        }
    } else {
        UC_ERROR("Unsupported transport operation op_type={}", static_cast<int>(ctx.opType));
        return Status::Error(StatusCode::UNSUPPORTED, "transport operation is unsupported");
    }
    return Status::OK();
}

void TransportTaskExecutor::BuildSubBatchSendBuffers(
    std::vector<TransportSubBatchContext>& subBatchContexts,
    std::vector<TransProvider::SendIoBatch>& ioBatches)
{
    ioBatches.reserve(subBatchContexts.size());

    for (auto& subBatchContext : subBatchContexts) {
        ioBatches.push_back(TransProvider::SendIoBatch{
            subBatchContext.channel->GetConnection(),
            reinterpret_cast<void*>(subBatchContext.sendSge.device_addr),
            reinterpret_cast<void*>(subBatchContext.flagBuffer.device_addr),
            subBatchContext.sendSge.length});
    }
}

void TransportTaskExecutor::SendSubBatchBuffers(
    std::vector<TransportSubBatchContext>& subBatchContexts,
    const std::vector<TransProvider::SendIoBatch>& ioBatches)
{
    if (ioBatches.empty()) { return; }

    const auto kernelCount = GetSendCountAttr(config_.attrs, "kernel_count");
    const auto quietCount = GetSendCountAttr(config_.attrs, "quiet_count");
    const auto sendStatuses = transProvider_->Send(ioBatches, kernelCount, quietCount);
    if (sendStatuses.size() != ioBatches.size()) {
        const auto status = Status::Error(StatusCode::INTERNAL_ERROR,
                                          "transport send returned unexpected status count");
        UC_ERROR("Transport send returned unexpected status count expected={} actual={}",
                 ioBatches.size(), sendStatuses.size());
        for (auto& subBatchContext : subBatchContexts) {
            SetSubBatchSendFailed(subBatchContext, status);
            ReleaseSubBatchResources(subBatchContext);
        }
        return;
    }

    for (std::size_t index = 0; index < sendStatuses.size(); ++index) {
        auto& subBatchContext = subBatchContexts[index];
        const auto& subBatchStatus = sendStatuses[index];
        if (subBatchStatus.ok()) { continue; }

        UC_ERROR("Send sub-batch failed sub_batch_index={} cid={} code={} message={}", index,
                 subBatchContext.cid, static_cast<int>(subBatchStatus.code),
                 subBatchStatus.message);
        SetSubBatchSendFailed(subBatchContext, subBatchStatus);
        connManager_->ReportFailure(subBatchContext.channel);
        ReleaseSubBatchResources(subBatchContext);
    }
}

}  // namespace UC::ASU
