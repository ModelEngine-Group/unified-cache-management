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
#ifndef UNIFIEDCACHE_TRANS_STATUS_H
#define UNIFIEDCACHE_TRANS_STATUS_H

#include <fmt/format.h>
#include <string>

namespace UC::Trans {

class Status {
    static constexpr int32_t OK_ = 0;
    static constexpr int32_t ERROR_ = -1;
    int32_t code_;
    std::string message_;
    explicit Status(int32_t code) : code_(code) {}

public:
    bool operator==(const Status& other) const noexcept { return code_ == other.code_; }
    bool operator!=(const Status& other) const noexcept { return !(*this == other); }
    std::string ToString() const { return fmt::format("({}) {}", code_, message_); }
    constexpr bool Success() const noexcept { return code_ == OK_; }
    constexpr bool Failure() const noexcept { return !Success(); }

public:
    Status(int32_t code, std::string message) : code_{code}, message_{std::move(message)} {}
    static Status OK() { return Status{OK_}; }
    static Status Error(std::string message) { return {ERROR_, std::move(message)}; }
};

} // namespace UC::Trans

#endif
