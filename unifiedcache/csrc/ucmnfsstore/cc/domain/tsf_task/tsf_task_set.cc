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
#include "tsf_task_set.h"
#include <mutex>
#include <functional>
#include <algorithm>

namespace UC{

size_t TsfTaskSet::Hash(const size_t id)
{
    return id % nBucket;
}

void TsfTaskSet::Insert(const size_t id)
{
    size_t bucketId = Hash(id);
    std::unique_lock<std::shared_mutex> lock(this->_mutexs[bucketId]);
    this->_sets[bucketId].push_back(id);
}

bool TsfTaskSet::Exist(const size_t id)
{
    size_t bucketId = Hash(id);
    std::shared_lock<std::shared_mutex> lock(this->_mutexs[bucketId]);
    return std::find(this->_sets[bucketId].begin(), this->_sets[bucketId].end(), id) != this->_sets[bucketId].end();
}

void TsfTaskSet::Remove(const size_t id)
{
    size_t bucketId = Hash(id);
    std::unique_lock<std::shared_mutex> lock(this->_mutexs[bucketId]);
    this->_sets[bucketId].remove(id);
}

} // namespace UC