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
#ifndef UNIFIEDCACHE_TSF_TASK_RUNNER
#define UNIFIEDCACHE_TSF_TASK_RUNNER

#include <string>
#include "idevice.h"
#include "status/status.h"
#include "tsf_task.h"

namespace UC{

class TsfTaskRunner{
public:
    Status Run(const TsfTask& task, IDevice* device);
     
private:
    Status Host2SSD(const TsfTask& task);   
    Status SSD2Host(const TsfTask& task);   
    Status Device2SSD(const TsfTask& task); // todo
    Status SSD2Device(const TsfTask& task); // todo
    Status ReadFile(const TsfTask& task, bool align = false);
    Status WriteFile(const TsfTask& task, bool align = false); 
};

} // namespace UC

#endif