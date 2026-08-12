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
 */
#ifndef UNIFIEDCACHE_PIPELINE_STORE_HEALTH_CONFIG_H
#define UNIFIEDCACHE_PIPELINE_STORE_HEALTH_CONFIG_H

#include <chrono>
#include <cstddef>
#include "status/status.h"

namespace UC::PipelineStore {

struct StoreHealthConfig {
    bool enabled{false};
    std::chrono::milliseconds healthCheckInterval{std::chrono::seconds(10)};
    std::chrono::milliseconds healthCheckTimeout{std::chrono::seconds(3)};
    size_t healthWindowSize{8};
    size_t failureThreshold{2};

    Status Validate() const
    {
        if (healthCheckInterval.count() <= 0 || healthCheckTimeout.count() <= 0 ||
            healthWindowSize == 0 || failureThreshold == 0) {
            return Status::InvalidParam("store health values must be positive");
        }
        if (failureThreshold > healthWindowSize) {
            return Status::InvalidParam("failure threshold exceeds health window");
        }
        if (healthCheckTimeout >= healthCheckInterval) {
            return Status::InvalidParam("health timeout must be shorter than interval");
        }
        return Status::OK();
    }
};

}  // namespace UC::PipelineStore

#endif
