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
#include "transport_task_completion.h"
#include "buffer_manager.h"
#include "connection_internal.h"

namespace UC::ASU {

namespace {

bool IsSubBatchTerminal(TransportSubBatchState state)
{
    return state == TransportSubBatchState::COMPLETED || state == TransportSubBatchState::FAILED;
}

Status BuildTaskFinalStatus(const TransportTaskContext& ctx)
{
    for (const auto& subBatchContext : ctx.subBatchContexts) {
        if (!subBatchContext.status.ok()) {
            return Status::Error(StatusCode::PARTIAL_FAILED, "one or more sub-batches failed");
        }
    }

    if (!ctx.finalStatus.ok()) { return ctx.finalStatus; }
    return Status::OK();
}

TransportTaskState BuildTaskStateFromSubBatches(const TransportTaskContext& ctx)
{
    for (const auto& subBatchContext : ctx.subBatchContexts) {
        if (subBatchContext.state == TransportSubBatchState::FAILED) {
            return TransportTaskState::FAILED;
        }
    }
    return TransportTaskState::COMPLETED;
}

}  // namespace

void InitializeTerminalSubBatchCount(TransportTaskContext& ctx)
{
    // At submit completion time, terminal sub-batches are usually submit/send failures.
    ctx.completedSubBatchCount = 0;
    for (const auto& subBatchContext : ctx.subBatchContexts) {
        if (!IsSubBatchTerminal(subBatchContext.state)) { continue; }

        ++ctx.completedSubBatchCount;
    }
}

Status ReleaseSubBatchResources(TransportSubBatchContext& subBatchContext,
                                BufferManager& sendBufferManager, BufferManager& flagBufferManager)
{
    Status finalStatus = Status::OK();

    if (subBatchContext.sendSge.slot_index != UINT32_MAX) {
        auto status = sendBufferManager.Free(subBatchContext.sendSge.slot_index);
        if (!status.ok()) {
            if (finalStatus.ok()) { finalStatus = status; }
        }
        subBatchContext.sendSge = {};
    }

    if (subBatchContext.flagBuffer.slot_index != UINT32_MAX) {
        auto status = flagBufferManager.Free(subBatchContext.flagBuffer.slot_index);
        if (!status.ok()) {
            if (finalStatus.ok()) { finalStatus = status; }
        }
        subBatchContext.flagBuffer = {};
    }

    if (subBatchContext.channel != nullptr) {
        subBatchContext.channel->ReleaseInflight();
        subBatchContext.channel = nullptr;
    }

    return finalStatus;
}

Status ReleaseAllSubBatchResources(std::vector<TransportSubBatchContext>& subBatchContexts,
                                   BufferManager& sendBufferManager,
                                   BufferManager& flagBufferManager)
{
    Status finalStatus = Status::OK();
    for (auto& subBatchContext : subBatchContexts) {
        const auto status =
            ReleaseSubBatchResources(subBatchContext, sendBufferManager, flagBufferManager);
        if (finalStatus.ok() && !status.ok()) { finalStatus = status; }
    }
    return finalStatus;
}

void CompleteSubBatch(TransportTaskContext& ctx, TransportSubBatchContext& subBatchContext,
                      TransportSubBatchState state, const Status& status,
                      BufferManager& sendBufferManager, BufferManager& flagBufferManager)
{
    if (subBatchContext.state != TransportSubBatchState::PENDING) { return; }

    const auto releaseStatus =
        ReleaseSubBatchResources(subBatchContext, sendBufferManager, flagBufferManager);
    const auto completionStatus = status.ok() ? releaseStatus : status;
    subBatchContext.state = (!completionStatus.ok() && state == TransportSubBatchState::COMPLETED)
                                ? TransportSubBatchState::FAILED
                                : state;
    subBatchContext.status = completionStatus;
    ++ctx.completedSubBatchCount;
}

void TryFinalizeTaskFromSubBatches(TransportTaskContext& ctx)
{
    if (ctx.subBatchContexts.empty()) {
        ctx.state.store(
            ctx.finalStatus.ok() ? TransportTaskState::COMPLETED : TransportTaskState::FAILED,
            std::memory_order_release);
        return;
    }

    if (ctx.completedSubBatchCount != static_cast<std::uint32_t>(ctx.subBatchContexts.size())) {
        return;
    }

    ctx.finalStatus = BuildTaskFinalStatus(ctx);
    ctx.state.store(BuildTaskStateFromSubBatches(ctx), std::memory_order_release);
}

}  // namespace UC::ASU
