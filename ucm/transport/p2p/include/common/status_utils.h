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

#pragma once

#include <utility>
#include "logger/logger.h"
#include "status/status.h"

namespace transport {

template <typename Cleanup>
class ScopeGuard {
public:
    explicit ScopeGuard(Cleanup cleanup) : cleanup_(std::move(cleanup)) {}
    ~ScopeGuard()
    {
        if (active_) { cleanup_(); }
    }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    void Dismiss() { active_ = false; }

private:
    Cleanup cleanup_;
    bool active_ = true;
};

template <typename Cleanup>
ScopeGuard<Cleanup> MakeScopeGuard(Cleanup cleanup)
{
    return ScopeGuard<Cleanup>(std::move(cleanup));
}

}  // namespace transport

#define P2P_RETURN_IF_ERROR(expression, ...)                                \
    do {                                                                    \
        const auto transport_status = (expression);                         \
        if (transport_status.Failure()) {                                   \
            UC_ERROR("{}: {}", fmt::format(__VA_ARGS__), transport_status); \
            return transport_status;                                        \
        }                                                                   \
    } while (false)

#define P2P_RETURN_IF_TRUE(condition, error, ...)                           \
    do {                                                                    \
        const bool transport_condition = (condition);                       \
        if (transport_condition) {                                          \
            const auto transport_status = (error);                          \
            UC_ERROR("{}: {}", fmt::format(__VA_ARGS__), transport_status); \
            return transport_status;                                        \
        }                                                                   \
    } while (false)

#define P2P_LOG_IF_ERROR(expression, ...)                                   \
    do {                                                                    \
        const auto transport_status = (expression);                         \
        if (transport_status.Failure()) {                                   \
            UC_ERROR("{}: {}", fmt::format(__VA_ARGS__), transport_status); \
        }                                                                   \
    } while (false)
