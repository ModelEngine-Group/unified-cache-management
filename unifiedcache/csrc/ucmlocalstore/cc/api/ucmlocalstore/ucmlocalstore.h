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
#ifndef UCM_LOCAL_STORE_LOCAL_STORE_H
#define UCM_LOCAL_STORE_LOCAL_STORE_H

#include <list>
#include <string>

namespace UCM {

struct SetupParam {
    size_t capacity;
    size_t cacheSize;
    int32_t deviceId;
    size_t ioSize;

    SetupParam(const size_t capacity, const size_t cacheSize)
        : capacity{capacity}, cacheSize{cacheSize}, deviceId{-1}, ioSize{262144}
    {
    }
};

int32_t Setup(const SetupParam& param);
std::list<bool> LookupBatch(const std::list<std::string>& blockIdList);
size_t SubmitRead(const std::list<std::string>& blockIdList, const std::list<uintptr_t> dstList,
        const std::list<size_t> lengthList, const std::list<size_t> offsetList);
size_t SubmitWrite(const std::list<std::string>& blockIdList, const std::list<uintptr_t> srcList);
int32_t Wait(const size_t cacheId);

} // namespace UCM

#endif
