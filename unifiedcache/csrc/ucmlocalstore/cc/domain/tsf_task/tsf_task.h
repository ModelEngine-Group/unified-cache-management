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
#ifndef UCM_LOCAL_STORE_TSF_TASK_H
#define UCM_LOCAL_STORE_TSF_TASK_H

#include <memory>
#include "tsf_task_waiter.h"

namespace UCM {

class TsfTask {

public:
    TsfTask(const std::string& blockId, const size_t offset, const uintptr_t address, const size_t length)
        : blockId{blockId}, offset{offset}, address{address}, length{length}, buffer{nullptr},
          owner{0}, waiter{nullptr}, hub{nullptr}
    {
    }
    TsfTask() : TsfTask{{}, 0, 0, 0} {}

public:
    std::string blockId;
    size_t offset;
    uintptr_t address;
    size_t length;
    void* buffer;

    size_t owner;
    std::shared_ptr<TsfTaskWaiter> waiter;
    std::shared_ptr<std::byte> hub;
};

} // namespace UCM

#endif
