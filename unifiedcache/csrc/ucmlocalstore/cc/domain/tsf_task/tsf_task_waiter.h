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
#ifndef UCM_LOCAL_STORE_TSF_TASK_WAITER_H
#define UCM_LOCAL_STORE_TSF_TASK_WAITER_H

#include <spdlog/stopwatch.h>
#include "logger/logger.h"
#include "thread/latch.h"

namespace UCM {

class TsfTaskWaiter : public Latch {
public:
    TsfTaskWaiter(const size_t id, const size_t size, const size_t number)
        : Latch{number}, _id{id}, _size{size}, _number{number}
    {
    }

    void Done()
    {
        if (Latch::Done() == 0) {
            auto elapsed = this->_sw.elapsed().count();
            UCM_INFO("Cache Task({},{},{}) finished, elapsed={:.06f}s, bw={:.06f}GB/s.", this->_id,
                    this->_number, this->_size, elapsed, this->_size / elapsed / (1ULL << 30));
            this->Notify();
        }
    }


private:
    size_t _id;
    size_t _size;
    size_t _number;
    spdlog::stopwatch _sw;
};

} // namespace UCM

#endif // UCM_LOCAL_STORE_TSF_TASK_WAITER_H