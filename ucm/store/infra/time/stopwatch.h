/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_STOPWATCH_H
#define UNIFIEDCACHE_STOPWATCH_H

#include <chrono>

namespace UC {

class StopWatch {
    using clock = std::chrono::steady_clock;
    std::chrono::time_point<clock> startTp_;

public:
    StopWatch() : startTp_{clock::now()} {}
    std::chrono::duration<double> Elapsed() const
    {
        return std::chrono::duration<double>(clock::now() - startTp_);
    }
    std::chrono::milliseconds ElapsedMs() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - startTp_);
    }
    void Reset() { startTp_ = clock::now(); }
};

} // namespace UC

#endif
