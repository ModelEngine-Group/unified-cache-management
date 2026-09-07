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

#include <optional>
#include <string>
#include <utility>
#include "src/protocol/ub_error_code.h"

namespace umc::comm {

class UbStatus {
public:
    UbStatus() = default;
    UbStatus(UbErrorCode code, std::string msg = {}) : code_(code), msg_(std::move(msg)) {}

    static UbStatus Ok() { return UbStatus(UbErrorCode::Ok); }

    bool IsOk() const { return code_ == UbErrorCode::Ok; }
    bool IsError() const { return !IsOk(); }
    UbErrorCode Code() const { return code_; }
    const std::string& Message() const { return msg_; }

    explicit operator bool() const { return IsOk(); }

    UbStatus WithContext(std::string ctx) const
    {
        if (IsOk()) return *this;
        std::string m = ctx;
        if (!msg_.empty()) {
            m += ": ";
            m += msg_;
        }
        return UbStatus(code_, std::move(m));
    }

private:
    UbErrorCode code_{UbErrorCode::Ok};
    std::string msg_;
};

template <class T>
class Result {
public:
    Result(T value) : value_(std::move(value)) {}
    Result(UbStatus err) : status_(std::move(err)) {}

    bool IsOk() const { return status_.IsOk(); }
    const UbStatus& Status() const { return status_; }
    const T& operator*() const { return *value_; }
    T& operator*() { return *value_; }
    const T* operator->() const { return &(*value_); }
    T* operator->() { return &(*value_); }

private:
    std::optional<T> value_;
    UbStatus status_;
};

const char* UbErrorCodeToString(UbErrorCode code);

#define UB_RETURN_IF_ERROR(expr)           \
    do {                                   \
        ::umc::comm::UbStatus _s = (expr); \
        if (_s.IsError()) return _s;       \
    } while (0)

#define UB_RETURN_IF_ERROR_CTX(expr, ctx)             \
    do {                                              \
        ::umc::comm::UbStatus _s = (expr);            \
        if (_s.IsError()) return _s.WithContext(ctx); \
    } while (0)

}  // namespace umc::comm
