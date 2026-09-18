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
#ifndef COPY_INSTANCE_H
#define COPY_INSTANCE_H

#include <chrono>
#include <utility>
#include <vector>
#include "copy_buffer.h"
#include "copy_result.h"

class CopyInstance {
protected:
    size_t iterations_;
    bool affinitySrc_;

    virtual size_t AffinityDeviceId(const CopyBuffer& src, const CopyBuffer& dst) const
    {
        return (affinitySrc_ ? src : dst).Device();
    }

    virtual void Prepare(const std::vector<const CopyBuffer*>& srcBuffers,
                         const std::vector<const CopyBuffer*>& dstBuffers) = 0;

    virtual std::pair<size_t, size_t> DoCopyOnce() = 0;

    virtual void Cleanup() = 0;

public:
    CopyInstance(size_t iterations, bool affinitySrc)
        : iterations_(iterations), affinitySrc_(affinitySrc)
    {
    }

    virtual ~CopyInstance() = default;

    virtual std::string Name() const = 0;

    CopyResult::Result DoCopyBatch(const std::vector<const CopyBuffer*>& srcBuffers,
                                   const std::vector<const CopyBuffer*>& dstBuffers)
    {
        Prepare(srcBuffers, dstBuffers);

        constexpr int warmup = 3;
        for (int i = 0; i < warmup; i++) { DoCopyOnce(); }

        std::vector<size_t> submitCostArray;
        std::vector<size_t> copyCostArray;
        for (size_t i = 0; i < iterations_; i++) {
            auto [copyCost, submitCost] = DoCopyOnce();
            copyCostArray.push_back(copyCost);
            submitCostArray.push_back(submitCost);
        }

        Cleanup();

        return {srcBuffers.front()->Name(),
                dstBuffers.front()->Name(),
                Name(),
                srcBuffers.front()->Size(),
                srcBuffers.front()->Number() * srcBuffers.size(),
                std::move(submitCostArray),
                std::move(copyCostArray)};
    }

    CopyResult::Result DoCopy(const CopyBuffer* srcBuffer, const CopyBuffer* dstBuffer)
    {
        return DoCopyBatch({srcBuffer}, {dstBuffer});
    }
};

#endif  // COPY_INSTANCE_H
