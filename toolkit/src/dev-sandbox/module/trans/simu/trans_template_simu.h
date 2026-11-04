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
#ifndef TRANS_TEMPLATE_SIMU_H
#define TRANS_TEMPLATE_SIMU_H

#include <cstring>
#include <thread>
#include "trans_stopwatch.h"
#include "trans_template.h"

class TransMemcpyTemplate : public TransTemplate {
protected:
    struct TransTask {
        const TransBuffer* src;
        const TransBuffer* dst;
    };
    std::vector<TransTask> tasks_;

    void OnTransPre(const std::vector<const TransBuffer*>& srcBuffers,
                    const std::vector<const TransBuffer*>& dstBuffers) override
    {
        tasks_.clear();
        for (size_t i = 0; i < srcBuffers.size(); i++) {
            tasks_.push_back({srcBuffers[i], dstBuffers[i]});
        }
    }
    std::pair<size_t, size_t> OnTrans() override
    {
        TransStopwatch submitWatch;
        std::vector<std::thread> threads;
        threads.reserve(tasks_.size());
        TransStopwatch execWatch;
        for (auto& task : tasks_) {
            threads.emplace_back([&task]() {
                for (size_t j = 0; j < task.src->Number(); j++) {
                    std::memcpy((*task.dst)[j], (*task.src)[j], task.src->Size());
                }
            });
        }
        size_t submitCost = submitWatch.Elapse();
        for (auto& t : threads) { t.join(); }
        size_t execCost = execWatch.Elapse();
        return {execCost, submitCost};
    }
    void OnTransPost() override { tasks_.clear(); }

public:
    TransMemcpyTemplate(size_t iterations, bool affinitySrc)
        : TransTemplate{iterations, affinitySrc}
    {
    }
    std::string Name() const override { return "memcpy"; }
};

#endif
