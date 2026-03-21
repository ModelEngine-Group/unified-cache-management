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
#ifndef UNIFIEDCACHE_INFRA_CPU_AFFINITY_H
#define UNIFIEDCACHE_INFRA_CPU_AFFINITY_H

#include <sched.h>
#include <thread>
#include <vector>
#include "status/status.h"

namespace UC {

class CpuAffinity {
public:
    static Status SetCpuAffinity4CurrentThread(const cpu_set_t& mask)
    {
        if (CPU_COUNT(&mask) == 0) { return Status::InvalidParam(); }
        auto ret = sched_setaffinity(0, sizeof(mask), &mask);
        if (ret != 0) { return Status::Error(std::to_string(errno)); }
        std::this_thread::yield();
        return Status::OK();
    }
    static Status SetCpuAffinity4CurrentThread(const std::vector<ssize_t> cores)
    {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        for (const auto core : cores) { CPU_SET(core, &mask); }
        return SetCpuAffinity4CurrentThread(mask);
    }
};

}  // namespace UC

#endif  // UNIFIEDCACHE_INFRA_CPU_AFFINITY_H
