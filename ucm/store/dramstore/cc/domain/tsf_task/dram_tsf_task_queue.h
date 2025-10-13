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
#ifndef UNIFIEDCACHE_DRAM_TSF_TAKS_QUEUE_H
#define UNIFIEDCACHE_DRAM_TSF_TAKS_QUEUE_H

#include "idevice.h"
#include "space/dram_space_layout.h"
#include "thread/thread_pool.h"
#include "dram_tsf_task.h"
#include "dram_tsf_task_set.h"

namespace UC {

class DramTsfTaskQueue {
public:
    Status Setup(const int32_t deviceId, 
                //  const size_t bufferSize, const size_t bufferNumber,
                 DramTsfTaskSet* failureSet, const DramSpaceLayout* layout); // TODO：需要传入哪些参数，到时候再看。目前看需要deviceId，但是bufferSize和bufferNumber不需要
    void Push(std::list<DramTsfTask>& tasks);

private:
    void StreamOper(DramTsfTask& task);
    // void FileOper(DramTsfTask& task); // 与下面同步，不再需要了
    void H2D(DramTsfTask& task); // 这个函数和D2H函数里，需要用到this->_layout里的接口来得到数据在host内存中的位置，再调用this->_device来执行显卡与host中的内存拷贝的操作
    void D2H(DramTsfTask& task); // 完成后，要更新_layout中的_dataStoreMap字典
    // void H2S(DramTsfTask& task);
    // void S2H(DramTsfTask& task);
    void Done(const DramTsfTask& task, bool success);

private:
    ThreadPool<DramTsfTask> _streamOper;
    // ThreadPool<DramTsfTask> _fileOper; // 在NFSStore中需要，但是在DRAM里不需要了，因为只涉及H与D之间的数据传输
    std::unique_ptr<IDevice> _device;
    DramTsfTaskSet* _failureSet;
    // TODO
    DramSpaceLayout* _layout; // 这个大概也需要吧，因为NFSStore里是在H2S和S2H中需要用它，也就是真正和“存储”打交道时是需要它的。具体怎么定义后面再看
};

} // namespace UC

#endif
