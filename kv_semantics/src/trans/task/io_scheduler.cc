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
#include "io_scheduler.h"
#include <algorithm>

namespace UC::ASU {

namespace {

std::size_t GetSubBatchCount(std::size_t total, std::size_t ioNum)
{
    return 1 + (total - 1) / ioNum;
}

template <typename Value, typename ScheduledBatch, typename SetView>
std::vector<ScheduledBatch> SplitBatchView(const BatchView<Value>& view, std::size_t ioNum,
                                           SetView setView)
{
    std::vector<ScheduledBatch> result;
    const std::size_t subBatchCount = GetSubBatchCount(view.size, ioNum);
    result.reserve(subBatchCount);

    for (std::size_t offset = 0; offset < view.size; offset += ioNum) {
        const std::size_t end = std::min(offset + ioNum, view.size);

        auto& batch = result.emplace_back();
        setView(batch, BatchView<Value>{view.data + offset, end - offset});
    }

    return result;
}

}  // namespace

IoScheduler::IoScheduler(const TransportConfig& config)
    : batchLoadIoNum_(config.asuBatchLoadIoNum),
      batchStoreIoNum_(config.asuBatchStoreIoNum),
      deleteIoNum_(config.asuDeleteIoNum),
      queryIoNum_(config.asuQueryIoNum)
{
}

std::vector<IoScheduler::ScheduledIoBatch> IoScheduler::SplitForAsu(
    const BatchView<KVBuffer>& entries, AsuOpType opType) const
{
    return SplitBatchView<KVBuffer, ScheduledIoBatch>(
        entries, GetSqeIoNum(opType),
        [](ScheduledIoBatch& batch, BatchView<KVBuffer> view) { batch.entries = view; });
}

std::vector<IoScheduler::ScheduledKeyBatch> IoScheduler::SplitForAsu(
    const BatchView<CacheKey>& keys, AsuOpType opType) const
{
    return SplitBatchView<CacheKey, ScheduledKeyBatch>(
        keys, GetSqeIoNum(opType),
        [](ScheduledKeyBatch& batch, BatchView<CacheKey> view) { batch.keys = view; });
}

std::size_t IoScheduler::GetSqeIoNum(AsuOpType opType) const
{
    if (opType == AsuOpType::LOAD || opType == AsuOpType::STORE) { return 1; }
    switch (opType) {
        case AsuOpType::BATCH_LOAD: return batchLoadIoNum_;
        case AsuOpType::BATCH_STORE: return batchStoreIoNum_;
        case AsuOpType::DELETE: return deleteIoNum_;
        case AsuOpType::QUERY: return queryIoNum_;
        default: return 0;
    }
}

}  // namespace UC::ASU
