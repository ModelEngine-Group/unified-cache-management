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

std::size_t GetSubBatchCount(std::size_t total, std::size_t maxIoSize)
{
    if (total == 0 || maxIoSize == 0) { return 0; }
    return 1 + (total - 1) / maxIoSize;
}

}  // namespace

std::vector<IoScheduler::ScheduledIoBatch> IoScheduler::SplitForAsu(
    const BatchView<KVBuffer>& entries, std::size_t maxIoSize) const
{
    std::vector<ScheduledIoBatch> result;
    if (entries.empty() || maxIoSize == 0) { return result; }

    const std::size_t subBatchCount = GetSubBatchCount(entries.size, maxIoSize);
    result.reserve(subBatchCount);

    for (std::size_t offset = 0; offset < entries.size; offset += maxIoSize) {
        const std::size_t end = std::min(offset + maxIoSize, entries.size);

        ScheduledIoBatch batch;
        batch.entries = BatchView<KVBuffer>{entries.data + offset, end - offset};
        result.push_back(batch);
    }

    return result;
}

std::vector<IoScheduler::ScheduledKeyBatch> IoScheduler::SplitForAsu(
    const BatchView<CacheKey>& keys, std::size_t maxIoSize) const
{
    std::vector<ScheduledKeyBatch> result;
    if (keys.empty() || maxIoSize == 0) { return result; }

    const std::size_t subBatchCount = GetSubBatchCount(keys.size, maxIoSize);
    result.reserve(subBatchCount);

    for (std::size_t offset = 0; offset < keys.size; offset += maxIoSize) {
        const std::size_t end = std::min(offset + maxIoSize, keys.size);

        ScheduledKeyBatch batch;
        batch.keys = BatchView<CacheKey>{keys.data + offset, end - offset};
        result.push_back(batch);
    }

    return result;
}

std::size_t GetSqeMaxIoSize(TransportOpType opType)
{
    if (opType == TransportOpType::LOAD || opType == TransportOpType::STORE) { return 1; }
    return GetAsuMaxIoSize(opType);
}

}  // namespace UC::ASU
