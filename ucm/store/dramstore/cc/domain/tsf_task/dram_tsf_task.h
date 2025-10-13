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
#ifndef UNIFIEDCACHE_DRAM_TSF_TASK_H
#define UNIFIEDCACHE_DRAM_TSF_TASK_H

#include "ucmstore.h"
#include "dram_tsf_task_waiter.h"

namespace UC {

class DramTsfTask {
public:
    using Type = CCStore::Task::Type;
    // using Location = CCStore::Task::Location; // 这个不再需要了，因为只有D2H和H2D两种传输

public:
    DramTsfTask(const Type type, const std::string& blockId,
            const size_t offset, const uintptr_t address, const size_t length)
        : type{type}, blockId{blockId}, offset{offset}, address{address},
          length{length}, owner{0}, waiter{nullptr}
    {
    }
    DramTsfTask() : DramTsfTask{Type::DUMP, {}, 0, 0, 0} {}

public:
    Type type;
    // Location location; // 不需要了
    std::string blockId; // 对于一个task来说，这个bloockID和下一行的offset还是需要的，因为它们本质上是上层传来的。（参考dramstore.py.cc）
    size_t offset;
    uintptr_t address; // 在显卡上的地址
    size_t length; // 数据传输的长度

    size_t owner; // 大的Task的TaskId
    std::shared_ptr<DramTsfTaskWaiter> waiter;
    // std::shared_ptr<std::byte> hub; // 在nfsstore中，这个的意思是中转站（对于nfsstore来说，host的目的就是数据中转），因此在dram_connector中不再需要
};

} // namespace UC

#endif
