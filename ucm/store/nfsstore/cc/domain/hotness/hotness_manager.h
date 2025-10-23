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

#ifndef UNIFIEDCACHE_HOTNESS_MANAGER_H
#define UNIFIEDCACHE_HOTNESS_MANAGER_H

#include <atomic>
#include <functional>
#include "hotness_set.h"
#include "hotness_timer.h"
#include "logger/logger.h"

namespace UC {

class HotnessManager {
public:
    Status Setup(const size_t interval, const SpaceLayout* spaceLayout)
    {
        this->hotnessTimer_.SetInterval(interval);
        this->layout_ = spaceLayout;
        this->setupSuccess_ = true;
        return Status::OK();
    }
    
    void Visit(const std::string& blockId)
    {
        if (!this->setupSuccess_) {
            return;
        }

        this->hotnessSet_.Insert(blockId);
        auto old = this->serviceRunning_.load(std::memory_order_acquire);
        if (old) { return; }
        if (this->serviceRunning_.compare_exchange_weak(old, true, std::memory_order_acq_rel)) {
            auto updater = std::bind(&HotnessSet::UpdateHotness, &this->hotnessSet_, this->layout_);
            if (this->hotnessTimer_.Start(std::move(updater)).Success()) {
                UC_INFO("Space hotness service started.");
                return;
            }
            this->serviceRunning_ = old;
        }
    }

private:
    bool setupSuccess_{false};
    std::atomic_bool serviceRunning_{false};
    const SpaceLayout* layout_;
    HotnessSet hotnessSet_;
    HotnessTimer hotnessTimer_;
};

} // namespace UC

#endif