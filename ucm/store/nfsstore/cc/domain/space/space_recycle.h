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
#ifndef UNIFIEDCACHE_SPACE_RECYCLE_H
#define UNIFIEDCACHE_SPACE_RECYCLE_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include "space_layout.h"

namespace UC {

class SpaceRecycle {
public:
    using RecycleOneBlockDone = std::function<void(void)>;
    SpaceRecycle() = default;
    SpaceRecycle(const SpaceRecycle&) = delete;
    SpaceRecycle& operator=(const SpaceRecycle&) = delete;
    ~SpaceRecycle();
    Status Setup(const SpaceLayout* layout, const size_t totalNumber,
                 RecycleOneBlockDone done);
    void Trigger();
private:
    void Recycler();
private:
    bool stop_{false};
    bool recycling_{false};
    std::atomic_bool serviceRunning_{false};
    uint32_t recycleNum_{0};
    RecycleOneBlockDone recycleOneBlockDone_;
    const SpaceLayout* layout_{nullptr};
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread worker_;
};

} // namespace UC
#endif