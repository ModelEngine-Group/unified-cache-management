/**
 * MIT License
 *
 * Copyright (c) 2026 Mag1c.H
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
#ifndef COPY_INSTANCE_SIMU_H
#define COPY_INSTANCE_SIMU_H

#include <chrono>
#include <cstring>
#include <utility>
#include <vector>
#include "copy_buffer.h"
#include "copy_instance.h"
#include "error_handle.h"

struct SimuStreamContext {
    size_t deviceId;
    size_t size;
    std::vector<void*> src;
    std::vector<void*> dst;
};

class MemcpyCopyInstance : public CopyInstance {
protected:
    std::vector<SimuStreamContext> contexts_;

    void Prepare(const std::vector<const CopyBuffer*>& srcBuffers,
                 const std::vector<const CopyBuffer*>& dstBuffers) override
    {
        contexts_.clear();
        const auto bufferNumber = srcBuffers.size();
        for (size_t i = 0; i < bufferNumber; i++) {
            auto& src = *srcBuffers[i];
            auto& dst = *dstBuffers[i];
            ASSERT(src.Number() == dst.Number());
            ASSERT(src.Size() == dst.Size());

            SimuStreamContext ctx;
            ctx.deviceId = AffinityDeviceId(src, dst);
            ctx.size = src.Size();
            ctx.src.reserve(src.Number());
            ctx.dst.reserve(dst.Number());
            for (size_t j = 0; j < src.Number(); j++) {
                ctx.src.push_back(src[j]);
                ctx.dst.push_back(dst[j]);
            }
            contexts_.push_back(std::move(ctx));
        }
    }

    void Cleanup() override { contexts_.clear(); }

    std::pair<size_t, size_t> DoCopyOnce() override
    {
        using namespace std::chrono;
        auto submitStart = steady_clock::now();
        for (const auto& ctx : contexts_) {
            for (size_t i = 0; i < ctx.src.size(); i++) {
                std::memcpy(ctx.dst[i], ctx.src[i], ctx.size);
            }
        }
        auto copyCost = duration_cast<microseconds>(steady_clock::now() - submitStart).count();
        return {copyCost, copyCost};
    }

public:
    MemcpyCopyInstance(size_t iterations, bool affinitySrc) : CopyInstance(iterations, affinitySrc)
    {
    }

    std::string Name() const override { return "memcpy"; }
};

#endif  // COPY_INSTANCE_SIMU_H
