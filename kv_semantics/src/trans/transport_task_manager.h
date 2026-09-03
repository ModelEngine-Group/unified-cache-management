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
#include <memory>
#include <vector>
#include "asu_transport.h"
#include "buffer_manager.h"
#include "connection_manager.h"
#include "task_context.h"
#include "task_manager_base.h"

namespace UC::ASU {

struct TransportSubBatchContext {
    std::uint16_t cid{0};
    AsuOpType opType{AsuOpType::QUERY};
    TransportSubBatchState state{TransportSubBatchState::PENDING};
    Status status{Status::OK()};
    std::shared_ptr<ConnectionChannel> channel;
    bool useSeekControl{false};
    ScatterGatherEntry sendSge;
    ScatterGatherEntry flagBuffer;
    std::vector<Status> entryStatus;
};

class TransportTaskManager : public TaskManagerBase<TransportTask, TransportTaskState> {
public:
    TransportTaskManager() : TaskManagerBase(TransportTaskState::PENDING, "transport") {}

    void NotifyCompletion(const TransportTaskPtr& task);
    static void BuildResult(const TransportTask& task, TaskResult& result);
};

}  // namespace UC::ASU
