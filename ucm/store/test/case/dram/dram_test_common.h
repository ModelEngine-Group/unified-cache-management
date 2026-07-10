/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_TEST_DRAM_COMMON_H
#define UNIFIEDCACHE_TEST_DRAM_COMMON_H

#include <chrono>
#include <cstdint>
#include <memory>
#include "detail/types_helper.h"
#include "dram/cc/entry.h"

namespace UC::Test::Dram {

using Clock = std::chrono::system_clock;
using TimePoint = Clock::time_point;

using UC::DramStore::Entry;
using UC::DramStore::EntryPtr;
using UC::DramStore::EntryStatus;

inline EntryPtr MakeEntry(UC::Detail::BlockId key, uint32_t position = 0,
                          TimePoint lifeTimeout = TimePoint{},
                          EntryStatus status = EntryStatus::READY, uint32_t refCnt = 0,
                          TimePoint leaseTimeout = TimePoint{})
{
    auto e = std::make_shared<Entry>();
    e->key = key;
    e->position = position;
    e->lifeTimeout = lifeTimeout;
    e->status = status;
    e->refCnt = refCnt;
    e->leaseTimeout = leaseTimeout;
    return e;
}

inline UC::Detail::BlockId KeyFromHex(const char* hex)
{
    return UC::Test::Detail::TypesHelper::MakeBlockId(hex);
}

}  // namespace UC::Test::Dram

#endif
