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

#include <cstddef>
#include <cstdint>
#include <vector>
#include "kv_transport.h"
#include "trans_task_manager.h"

namespace kv {

class IoScheduler {
public:
    IoScheduler() = default;
    explicit IoScheduler(const TransportConfig& config);

    struct ScheduledIoBatch {
        BatchView<KVBuffer> entries;
    };

    struct ScheduledKeyBatch {
        BatchView<CacheKey> keys;
    };

    std::vector<ScheduledIoBatch> SplitForAsu(const BatchView<KVBuffer>& entries,
                                              AsuOpType opType) const;
    std::vector<ScheduledKeyBatch> SplitForAsu(const BatchView<CacheKey>& keys,
                                               AsuOpType opType) const;
    std::size_t GetSqeIoNum(AsuOpType opType) const;

private:
    std::size_t batchLoadIoNum_{110};
    std::size_t batchStoreIoNum_{110};
    std::size_t deleteIoNum_{254};
    std::size_t queryIoNum_{256};
};

}  // namespace kv
