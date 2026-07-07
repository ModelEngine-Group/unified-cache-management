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
#ifndef TRANS_TEMPLATE_H
#define TRANS_TEMPLATE_H

#include "trans_buffer.h"
#include "trans_result.h"

class TransTemplate {
protected:
    size_t iterations_;
    bool affinitySrc_;

    virtual size_t AffinityDeviceId(const TransBuffer& src, const TransBuffer& dst) const
    {
        return (affinitySrc_ ? src : dst).Device();
    }
    virtual void OnTransPre(const std::vector<const TransBuffer*>& srcBuffers,
                            const std::vector<const TransBuffer*>& dstBuffers) = 0;
    virtual std::pair<size_t, size_t> OnTrans() = 0;
    virtual void OnTransPost() = 0;

public:
    TransTemplate(size_t iterations, bool affinitySrc)
        : iterations_(iterations), affinitySrc_(affinitySrc)
    {
    }
    virtual ~TransTemplate() = default;
    virtual std::string Name() const = 0;
    TransResult::Result TransBatch(const std::vector<const TransBuffer*>& srcBuffers,
                                   const std::vector<const TransBuffer*>& dstBuffers)
    {
        OnTransPre(srcBuffers, dstBuffers);
        constexpr int warmup = 3;
        for (int i = 0; i < warmup; i++) { OnTrans(); }
        std::vector<size_t> submitCostArray;
        std::vector<size_t> copyCostArray;
        for (size_t i = 0; i < iterations_; i++) {
            auto [copyCost, submitCost] = OnTrans();
            copyCostArray.push_back(copyCost);
            submitCostArray.push_back(submitCost);
        }
        OnTransPost();
        return {srcBuffers.front()->Name(),
                dstBuffers.front()->Name(),
                Name(),
                srcBuffers.front()->Size(),
                srcBuffers.front()->Number() * srcBuffers.size(),
                std::move(submitCostArray),
                std::move(copyCostArray)};
    }
    TransResult::Result TransOne(const TransBuffer* srcBuffer, const TransBuffer* dstBuffer)
    {
        return TransBatch({srcBuffer}, {dstBuffer});
    }
};

#endif
