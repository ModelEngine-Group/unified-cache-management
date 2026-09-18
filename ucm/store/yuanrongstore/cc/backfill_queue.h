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
 */
#ifndef UNIFIEDCACHE_YUANRONG_STORE_CC_BACKFILL_QUEUE_H
#define UNIFIEDCACHE_YUANRONG_STORE_CC_BACKFILL_QUEUE_H

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "datasystem/kv_client.h"
#include "status/status.h"
#include "yuanrong_config.h"

namespace UC::YuanRongStore {

struct BackfillTask {
    std::vector<std::string> keys;
    std::vector<std::shared_ptr<void>> hostBuffers;
};

class BackfillQueue {
public:
    ~BackfillQueue();
    Status Setup(const Config& config, std::shared_ptr<datasystem::KVClient> kvClient);
    bool Submit(BackfillTask task);
    void Close();

private:
    void WorkerStage();
    void RunOne(BackfillTask& task);

    Config config_;
    std::shared_ptr<datasystem::KVClient> kvClient_;
    size_t queueDepth_{0};
    bool stop_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<BackfillTask> waiting_;
    std::vector<std::thread> workers_;
};

}  // namespace UC::YuanRongStore

#endif
