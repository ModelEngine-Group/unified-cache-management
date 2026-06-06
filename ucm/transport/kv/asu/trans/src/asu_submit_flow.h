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
#include <string>
#include <unordered_map>
#include <vector>
#include "connection_manager.h"
#include "io_scheduler.h"
#include "sqe_request.h"
#include "transport_task_manager.h"

namespace UC::ASU {

class BufferManager;
class ProtocolManager;

Status SubmitTaskRequests(const TransportTaskContext& ctx, const IoScheduler& ioScheduler,
                          const std::unordered_map<std::string, std::string>& attrs,
                          const SqeCidAllocator& allocateSqeCid, BufferManager& sendBufferManager,
                          BufferManager& flagBufferManager, ProtocolManager& protocolManager,
                          std::vector<TransportSubBatchContext>& subBatchContexts);

Status BuildSubBatchSendBuffers(std::vector<TransportSubBatchContext>& subBatchContexts,
                                std::vector<SendIoBatch>& ioBatches,
                                std::vector<std::size_t>& subBatchIndexes,
                                BufferManager& sendBufferManager, BufferManager& flagBufferManager);

Status SendSubBatchBuffers(std::vector<TransportSubBatchContext>& subBatchContexts,
                           const std::vector<SendIoBatch>& ioBatches,
                           const std::vector<std::size_t>& subBatchIndexes,
                           const std::unordered_map<std::string, std::string>& attrs,
                           ConnectionManager& connManager);

}  // namespace UC::ASU
