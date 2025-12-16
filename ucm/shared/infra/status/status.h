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
#ifndef UNIFIEDCACHE_INFRA_STATUS_H
#define UNIFIEDCACHE_INFRA_STATUS_H

#include <cstdint>
#include <fmt/format.h>
#include <string>
#include <variant>

namespace UC {

template <int32_t i>
static inline constexpr int32_t __MakeStatusCode()
{
    return -50000 - i;
}

class Status {
    static constexpr int32_t OK_ = 0;
    static constexpr int32_t ERROR_ = -1;
    static constexpr int32_t EPARAM_ = __MakeStatusCode<0>();
    static constexpr int32_t EOOM_ = __MakeStatusCode<1>();
    static constexpr int32_t EOSERROR_ = __MakeStatusCode<2>();
    static constexpr int32_t EDUPLICATE_ = __MakeStatusCode<3>();
    static constexpr int32_t ERETRY_ = __MakeStatusCode<4>();
    static constexpr int32_t ENOOBJ_ = __MakeStatusCode<5>();
    static constexpr int32_t ESERIALIZE_ = __MakeStatusCode<6>();
    static constexpr int32_t EDESERIALIZE_ = __MakeStatusCode<7>();
    static constexpr int32_t EUNSUPPORTED_ = __MakeStatusCode<8>();
    static constexpr int32_t ENOSPACE_ = __MakeStatusCode<9>();
    static constexpr int32_t ETIMEOUT_ = __MakeStatusCode<10>();
    int32_t code_;
    std::string message_;
    explicit Status(int32_t code) : code_(code) {}

public:
    bool operator==(const Status& other) const noexcept { return code_ == other.code_; }
    bool operator!=(const Status& other) const noexcept { return !(*this == other); }
    int32_t Underlying() const { return code_; }
    std::string ToString() const
    {
        auto str = std::to_string(code_);
        if (message_.empty()) { return str; }
        return fmt::format("{}, {}", str, message_);
    }
    constexpr bool Success() const noexcept { return code_ == OK_; }
    constexpr bool Failure() const noexcept { return !Success(); }

public:
    Status(int32_t code, std::string message) : code_{code}, message_{std::move(message)} {}
    static Status OK() { return Status{OK_}; }
    static Status Error(std::string message) { return {ERROR_, std::move(message)}; }
    static Status Error() { return Status{ERROR_}; }
    static Status InvalidParam() { return Status{EPARAM_}; }
    static Status InvalidParam(std::string message) { return {EPARAM_, std::move(message)}; }
    template <typename... Args>
    static Status InvalidParam(fmt::format_string<Args...> fmt, Args&&... args)
    {
        return InvalidParam(fmt::format(fmt, std::forward<Args>(args)...));
    }
    static Status OutOfMemory() { return Status{EOOM_}; }
    static Status OsApiError() { return Status{EOSERROR_}; }
    static Status OsApiError(std::string message) { return Status{EOSERROR_, std::move(message)}; }
    static Status DuplicateKey() { return Status{EDUPLICATE_}; }
    static Status Retry() { return Status{ERETRY_}; }
    static Status NotFound() { return Status{ENOOBJ_}; }
    static Status SerializeFailed() { return Status{ESERIALIZE_}; }
    static Status DeserializeFailed() { return Status{EDESERIALIZE_}; }
    static Status Unsupported() { return Status{EUNSUPPORTED_}; }
    static Status NoSpace() { return Status{ENOSPACE_}; }
    static Status Timeout() { return Status{ETIMEOUT_}; }
};

template <class T>
class Expected {
    std::variant<Status, T> v_;

public:
    Expected(T&& val) : v_(std::move(val)) {}
    Expected(Status err) : v_(err) {}
    bool HasValue() const noexcept { return v_.index() == 1; }
    explicit operator bool() const noexcept { return HasValue(); }
    T& Value() & { return std::get<T>(v_); }
    T&& Value() && { return std::get<T>(std::move(v_)); }
    Status Error() const { return std::get<Status>(v_); }
};

inline std::string format_as(const Status& status) { return status.ToString(); }

}  // namespace UC

#endif
