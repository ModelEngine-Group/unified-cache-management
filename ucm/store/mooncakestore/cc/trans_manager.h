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
#ifndef UNIFIEDCACHE_MOONCAKE_STORE_CC_TRANS_MANAGER_H
#define UNIFIEDCACHE_MOONCAKE_STORE_CC_TRANS_MANAGER_H

#include <memory>
#include "dump_queue.h"
#include "global_config.h"
#include "host_buffer_pool.h"
#include "load_queue.h"
#include "real_client.h"
#include "share_load_queue.h"
#include "template/task_wrapper.h"
#include "trans_task.h"
#include "type/types.h"

namespace UC::MooncakeStore {

class TransManager : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
public:
    Status Setup(const Config& config);
    void Close();
    std::shared_ptr<mooncake::RealClient> GetRealClient() const { return realClient_; }

protected:
    void Dispatch(TaskPtr t, WaiterPtr w) override;

private:
    Status SetupRealClient(const Config& config);

    std::shared_ptr<mooncake::RealClient> realClient_;
    HostBufferPool bufPool_;
    LoadQueue loadQ_;
    DumpQueue dumpQ_;
    ShareLoadQueue shareLoadQ_;
    size_t localRankSize_{1};
    Config config_;
};

}  // namespace UC::MooncakeStore

#endif
