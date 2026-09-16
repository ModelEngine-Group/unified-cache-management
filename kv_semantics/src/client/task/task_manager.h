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
#include "task_context.h"
#include "task_manager_base.h"

namespace kv {

class ClientTaskManager : public TaskManagerBase<ClientTask, ClientTaskState> {
public:
    ClientTaskManager() : TaskManagerBase(ClientTaskState::PENDING, "client") {}

    bool Check(TaskId taskId);
    Status Wait(TaskId taskId, std::uint64_t waitTimeoutMs, TaskResult& result);
    Status Drain(std::uint64_t waitTimeoutMs);
    Status Process(const ClientTaskPtr& task);

    static void CompleteWithError(const ClientTaskPtr& task, const Status& status);
    static void CompleteTransportTask(const ClientTaskPtr& task, std::size_t transportTaskIndex,
                                      TaskResult result);
    static void CompleteUndispatchedTransportTasks(const ClientTaskPtr& task,
                                                   std::size_t firstTransportTaskIndex,
                                                   const Status& dispatchStatus);
    static void Finalize(const ClientTaskPtr& task);

private:
    static Status BuildTransportTasks(const ClientTaskPtr& task);
    static Status DispatchTask(const ClientTaskPtr& task);
    static Status BuildResult(const ClientTaskPtr& task, TaskResult& result);
    static Status WaitContext(const ClientTaskPtr& task, std::uint64_t waitTimeoutMs,
                              TaskResult& result);
};

}  // namespace kv
