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
#ifndef UNIFIEDCACHE_TRANSPORT_FFTS_DETAIL_FFTS_ENGINE_H
#define UNIFIEDCACHE_TRANSPORT_FFTS_DETAIL_FFTS_ENGINE_H

#include <cstddef>
#include <memory>
#include <vector>

#include <acl/acl.h>
#include <runtime/rt_ffts_plus.h>

#include "status/status.h"
#include "ffts_transport.h"

namespace UC::Transport::Ffts {

class FftsEngine {
public:
    FftsEngine();
    ~FftsEngine();

    // Non-copyable and non-movable
    FftsEngine(const FftsEngine&) = delete;
    FftsEngine& operator=(const FftsEngine&) = delete;
    FftsEngine(FftsEngine&&) = delete;
    FftsEngine& operator=(FftsEngine&&) = delete;

    Status Setup(int32_t deviceId);
    Status WaitEvent(void* event);
    Status Submit(const CopyDesc* copies, size_t count);
    Status Synchronize();

private:
    using ContextBuffer = std::vector<rtFftsPlusComCtx_t>;

    Status EnsureReady() const;
    Status SubmitChunk(const CopyDesc* copies, size_t count);
    void KeepAlive(std::shared_ptr<ContextBuffer> contexts);
    void ClearCompletedGraphs();

    int32_t deviceId_{-1};
    aclrtStream stream_{nullptr};
    bool ready_{false};
    std::vector<std::shared_ptr<ContextBuffer>> pendingContexts_;
};

}  // namespace UC::Transport::Ffts

#endif  // UNIFIEDCACHE_TRANSPORT_FFTS_DETAIL_FFTS_ENGINE_H
