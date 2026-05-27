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
#include "completion_poller.h"
#include <chrono>
#include <mutex>
#include "logger.h"

namespace UC::ASU {

namespace {

std::uint64_t CurrentMs()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace

CompletionPoller::~CompletionPoller() { Stop(); }

void CompletionPoller::Start(std::size_t queueCapacity, ReportFailureFunc reportFn, ReleaseFlagBufferFunc releaseFn)
{
    if (pollerThread_.joinable()) { return; }
    reportFailureFn_ = std::move(reportFn);
    releaseFlagBufferFn_ = std::move(releaseFn);
    pendingQueue_.Setup(queueCapacity);
    stop_.store(false, std::memory_order_release);
    pollerThread_ = std::thread(&CompletionPoller::PollLoop, this);
}

void CompletionPoller::Stop()
{
    if (!pollerThread_.joinable()) { return; }
    stop_.store(true, std::memory_order_release);
    if (pollerThread_.joinable()) { pollerThread_.join(); }

    PendingRequest req;
    while (pendingQueue_.TryPop(req)) {
        Finalize(req, Status::Error(StatusCode::CANCELED, "completion poller stopped"));
    }
}

void CompletionPoller::SubmitPending(PendingRequest req)
{
    pendingQueue_.Push(std::move(req));
}

void CompletionPoller::PollLoop()
{
    std::vector<PendingRequest> carry;
    while (!stop_.load(std::memory_order_acquire)) {
        std::vector<PendingRequest> batch;
        batch.swap(carry);

        PendingRequest req;
        while (pendingQueue_.TryPop(req)) {
            batch.push_back(std::move(req));
        }

        if (batch.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        for (auto& r : batch) {
            // 直接读取 flagBuffer 检查任务是否完成
            if (r.flagBuffer != nullptr) {
                volatile uint32_t* flagPtr = static_cast<volatile uint32_t*>(r.flagBuffer);
                if (*flagPtr != 0) {
                    Finalize(r, Status::OK());
                } else {
                    auto now = CurrentMs();
                    if (now >= r.deadlineMs) {
                        if (reportFailureFn_) { reportFailureFn_(r.channel); }
                        Finalize(r, Status::Error(StatusCode::TIMEOUT, "flag buffer timeout"));
                    } else {
                        carry.push_back(std::move(r));
                    }
                }
            } else {
                // 没有 flagBuffer，直接超时
                auto now = CurrentMs();
                if (now >= r.deadlineMs) {
                    Finalize(r, Status::Error(StatusCode::TIMEOUT, "no flag buffer"));
                } else {
                    carry.push_back(std::move(r));
                }
            }
        }
    }

    for (auto& r : carry) {
        Finalize(r, Status::Error(StatusCode::CANCELED, "completion poller stopped"));
    }
}

void CompletionPoller::Finalize(PendingRequest& req, Status status)
{
    auto& ctx = req.ctx;
    std::lock_guard<std::mutex> lock(ctx->waitMu);

    if (ctx->state.load(std::memory_order_acquire) == TransportTaskState::CANCELED) {
        req.channel->ReleaseInflight();
        // 释放 flagBuffer slot
        if (releaseFlagBufferFn_ && req.flagBuffer) {
            releaseFlagBufferFn_(req.flagBuffer);
        }
        return;
    }

    ctx->finalStatus = std::move(status);
    if (ctx->finalStatus.ok()) {
        if (ctx->opType == TransportOpType::QUERY) {
            ctx->queryResult.exists.assign(ctx->keys.size, 0);
            ctx->queryResult.prefixHitKeys = 0;
        }
        ctx->state.store(TransportTaskState::COMPLETED, std::memory_order_release);
    } else {
        ctx->state.store(TransportTaskState::FAILED, std::memory_order_release);
    }

    // 释放 flagBuffer slot
    if (releaseFlagBufferFn_ && req.flagBuffer) {
        releaseFlagBufferFn_(req.flagBuffer);
    }

    ctx->cv.notify_all();
    req.channel->ReleaseInflight();
}

}  // namespace UC::ASU
