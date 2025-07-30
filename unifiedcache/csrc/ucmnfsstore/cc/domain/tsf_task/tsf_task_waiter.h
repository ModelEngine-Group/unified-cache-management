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
#ifndef UNIFIEDCACHE_TSF_TASK_WAITER_H
#define UNIFIEDCACHE_TSF_TASK_WAITER_H

#include <spdlog/fmt/fmt.h>
#include <spdlog/stopwatch.h>
#include "logger/logger.h"
#include "thread/latch.h"

namespace UC {

class TsfTaskWaiter : public Latch {
public:
    TsfTaskWaiter(const size_t id, const size_t size, const size_t number, const std::string& brief) : Latch{0}
    {
        this->_ino = fmt::format("{}-{}-{}-{}", id, brief, size, number);
    }

    void Done()
    {
        if (Latch::Done() == 0) {
            UC_INFO("Task({}) finished in {:.006f}s", _ino, _sw.elapsed().count());
        }
    }


private:
    spdlog::stopwatch _sw;
    std::string _ino;
};

} // namespace UC

#endif // UNIFIEDCACHE_TSF_TASK_WAITER_H