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
 * DEALINGS IN THE SOFTWARE.
 * */
#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include "drampool_types.h"
#include "kv_protocol.h"
#include "status/status.h"

namespace UC::DramPool {

class TaskWorker final {
public:
    explicit TaskWorker(DramPoolRuntime& runtime);

    void Run(const std::atomic_bool& stop);

private:
    Status ProcessOneRequest(RequestTaskPtr task);
    Status ProcessDump(const KvDumpRequest& request, const transport::ManagerID& peerOneSidedId,
                       RequestTiming timing);
    Status ProcessLoad(const KvLoadRequest& request, const transport::ManagerID& peerOneSidedId,
                       RequestTiming timing);
    Status ProcessLookup(const KvLookupRequest& request, const transport::ManagerID& peerOneSidedId,
                         RequestTiming timing);

    void DeleteItemsMetadata(const std::vector<TransferItem>& items);
    void LoadEndItems(const std::vector<TransferItem>& items);
    Status QueueResponse(OpType opcode, std::uint64_t responseAddr,
                         const transport::ManagerID& peerOneSidedId,
                         std::vector<std::uint8_t>&& results, std::uint64_t requestId,
                         RequestTiming timing, bool dataTransferRequired, std::uint64_t dataBytes);
    Status SubmitCompletion(CompletionRecord&& record);

    DramPoolRuntime& runtime_;
};

}  // namespace UC::DramPool
