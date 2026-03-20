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
#include "hotness_tracker.h"
#include <utime.h>
#include "logger/logger.h"

namespace UC::PosixStore {

HotnessTracker::~HotnessTracker()
{
    stop_.store(true);
    if (utimeWorker_.joinable()) { utimeWorker_.join(); }
}

Status HotnessTracker::Setup(const SpaceLayout* layout)
{
    layout_ = layout;

    utimeWorker_ = std::thread(&HotnessTracker::UtimeWorkerLoop, this);
    return Status::OK();
}

void HotnessTracker::Touch(const Detail::BlockId& blockId)
{
    std::lock_guard<std::mutex> lock(queueMtx_);
    produceQueue_.push_back(blockId);
}

void HotnessTracker::UtimeWorkerLoop()
{
    std::deque<Detail::BlockId> consumeQueue;
    constexpr size_t kSpinLimit = 16;
    size_t spinCount = 0;
    while (!stop_.load()) {
        {
            std::lock_guard<std::mutex> lock(queueMtx_);
            consumeQueue.swap(produceQueue_);
        }
        if (consumeQueue.empty()) {
            if (++spinCount < kSpinLimit) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                spinCount = 0;
            }
            continue;
        }
        spinCount = 0;
        while (!consumeQueue.empty()) {
            auto filePath = layout_->DataFilePath(consumeQueue.front(), false);
            utime(filePath.c_str(), nullptr);
            consumeQueue.pop_front();
        }
    }
}

}  // namespace UC::PosixStore
