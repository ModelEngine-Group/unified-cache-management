/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_TRANS_MANAGER_H
#define UNIFIEDCACHE_POSIX_STORE_CC_TRANS_MANAGER_H

#include <shared_mutex>
#include "backend_manager.h"
#include "io_engine_aio.h"
#include "io_engine_psync.h"

namespace UC::PosixStore {

class TransManager {
    using IoEngine = Detail::TaskWrapper<TransTask, Detail::TaskHandle>;
    struct ShardTask {
        Detail::Shard shard;
        std::vector<std::string> attempted;
        size_t backendIndex{0};
        Status result{Status::OK()};
    };
    struct Request {
        explicit Request(TransTask source) : task{std::move(source)}
        {
            waiter.Set(task.desc.size());
        }
        TransTask task;
        double startTp{NowTime::Now()};
        // Each shard has one active attempt; waiter publishes all final results.
        std::vector<ShardTask> shardTasks;
        Latch waiter;
    };
    struct CallbackState {
        std::shared_mutex mutex;
        bool stopped{false};
    };

public:
    ~TransManager();
    Status Setup(const Config& config, const SpaceLayout* layout, const BackendManager* backendMgr);
    Expected<Detail::TaskHandle> Submit(TransTask task);
    Expected<bool> Check(Detail::TaskHandle handle);
    Status Wait(Detail::TaskHandle handle);

private:
    void TryNextBackend(const std::shared_ptr<Request>& request, ShardTask& shardTask);

    const BackendManager* backendMgr_{nullptr};
    size_t timeoutMs_{0};
    std::vector<std::unique_ptr<IoEngineAio>> aioEngines_;
    std::vector<std::unique_ptr<IoEnginePsync>> psyncEngines_;
    std::vector<IoEngine*> engines_;
    std::unordered_map<Detail::TaskHandle, std::shared_ptr<Request>> requests_;
    std::shared_mutex mutex_;
    std::shared_ptr<CallbackState> callbacks_{std::make_shared<CallbackState>()};
};

}  // namespace UC::PosixStore

#endif
