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
#ifndef UNIFIEDCACHE_TSF_TASK_SET
#define UNIFIEDCACHE_TSF_TASK_SET

#include <cstddef>
#include <shared_mutex>
#include <unordered_set>
#include <list>

namespace UC{

class TsfTaskSet{
    static constexpr size_t nBucket = 8192;
public:
    void Insert(const size_t id);
    bool Exist(const size_t id);
    void Remove(const size_t id);

private:
    size_t Hash(const size_t id);

private:
    std::shared_mutex _mutexs[nBucket];
    std::list<size_t> _sets[nBucket];
};

} // namespace UC

#endif