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
#include "status/status.h"
#include "type/block.h"
#include "type/token.h"

namespace UC::Proxy {

struct IoContext {
    /** Prerequisite handle the job's transfer waits on; 0 when there is none. */
    std::uintptr_t prerequisiteHandle{0};
};

struct IoResult {
    std::size_t jobId;
    bool success;
};

class StoreWorker {
public:
    virtual ~StoreWorker() = default;

    virtual Status SubmitPrefetch(std::size_t jobId, AddressedToken token) = 0;
    virtual Status SubmitGet(std::size_t jobId, std::vector<BlockPayload> payloads,
                             AddressedToken token, const IoContext& ctx) = 0;
    virtual Status SubmitPut(std::size_t jobId, std::vector<BlockPayload> payloads,
                             AddressedToken token, const IoContext& ctx) = 0;

    virtual std::vector<IoResult> GetFinished() = 0;
    virtual std::vector<IoResult> Wait(const std::vector<std::size_t>& jobIds) = 0;
};

}  // namespace UC::Proxy
