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
#ifndef UNIFIEDCACHE_TRANS_GDR_COPY_H
#define UNIFIEDCACHE_TRANS_GDR_COPY_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include "status/status.h"

enum GdrCopyKind : int {
    GdrMemcpyHostToDevice = 1,
    GdrMemcpyDeviceToHost = 2,
};

enum class GdrCompletionPollResult {
    Completed,
    Empty,
    Error,
    UnknownRequest,
};

class GdrCopyChannel {
public:
    virtual ~GdrCopyChannel() = default;

    virtual int GdrMemcpyAsync(void* dst, const void* src, size_t bytes, GdrCopyKind kind,
                               uint64_t* reqId) = 0;
    virtual int GdrMemcpyAsyncWithReqId(void* dst, const void* src, size_t bytes, GdrCopyKind kind,
                                        uint64_t reqId) = 0;
    virtual GdrCompletionPollResult PollCompletion(uint64_t* reqId) = 0;
    virtual int RequestCompletionNotification() = 0;
    virtual int WaitForCompletionEvent() = 0;
    virtual void InterruptCompletionWait() = 0;
};

class GdrCopyLib {
public:
    static std::shared_ptr<GdrCopyChannel> Open(int gpuId, const std::string& nicName);
    static void RegisterHostBuffer(void* host, size_t size);
    static void UnregisterHostBuffer(void* host);
    static UC::Status RegisterDeviceBuffer(void* device, size_t size);
    static void UnregisterDeviceBuffer(void* device);

private:
    GdrCopyLib() = delete;
};

#endif
