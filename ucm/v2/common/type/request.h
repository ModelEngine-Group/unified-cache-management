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
#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

namespace UC {

using RequestId = std::string;

/**
 * @brief Per-request facts about the context on whose behalf an
 *        operation is issued.
 *
 * @note Consumers read the fields they understand and ignore the
 *       rest; new facts are added as fields here instead of new
 *       parameters.
 * @note Facts are snapshotted at request registration and stay fixed
 *       for the request's lifetime; every window of the request sees
 *       the same values.
 */
struct RequestContext {
    /** Identity of the request these facts belong to. */
    RequestId requestId;
    /**
     * Predicted time-to-live of the requesting context; nullopt when
     * the control plane has no prediction (the store decides the
     * default, e.g. a configured fallback or never-expiring).
     * Prefetched blocks inherit the current window's request TTL.
     * Concurrent in-flight admissions of the same block are
     * serialized; the last writer's fact wins.
     */
    std::optional<std::chrono::milliseconds> ttl;
    /**
     * Predicted total length of the request, in blocks; nullopt when
     * the control plane has no prediction.
     */
    std::optional<std::uint64_t> totalBlockCount;
};

}  // namespace UC
