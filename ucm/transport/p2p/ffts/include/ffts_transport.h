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
#ifndef UNIFIEDCACHE_TRANSPORT_FFTS_TRANSPORT_H
#define UNIFIEDCACHE_TRANSPORT_FFTS_TRANSPORT_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "status/status.h"

namespace UC::Transport::Ffts {

struct CopyDesc {
    void* dst;
    const void* src;
    size_t size;
};

class FftsTransport {
public:
    FftsTransport();
    ~FftsTransport();

    FftsTransport(FftsTransport&&) noexcept;
    FftsTransport& operator=(FftsTransport&&) noexcept;

    FftsTransport(const FftsTransport&) = delete;
    FftsTransport& operator=(const FftsTransport&) = delete;

    Status Setup(int32_t deviceId);
    Status WaitEvent(void* event);
    Status CopyAsync(void* dst, const void* src, size_t size);
    Status Submit(const CopyDesc* copies, size_t count);
    Status Submit(const std::vector<CopyDesc>& copies);
    Status Synchronize();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace UC::Transport::Ffts

#endif  // UNIFIEDCACHE_TRANSPORT_FFTS_TRANSPORT_H
