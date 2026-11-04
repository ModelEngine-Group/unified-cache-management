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

#include <cstdint>
#include <fmt/format.h>
#include <string>
#include <type_traits>
#include <variant>

namespace UC {

#define UCM_STATUS_ERROR_LIST \
    X(Ok, 0)                  \
    X(General, -1)            \
    X(InvalidParam, -50000)   \
    X(OutOfMemory, -50001)    \
    X(OsApiError, -50002)     \
    X(DuplicateKey, -50003)   \
    X(Retry, -50004)          \
    X(NotFound, -50005)       \
    X(Unsupported, -50008)    \
    X(NoSpace, -50009)        \
    X(Timeout, -50010)        \
    X(Unhealthy, -50011)

class Status {
public:
    enum class Error : int32_t {
#define X(name, init) name = init,
        UCM_STATUS_ERROR_LIST
#undef X
    };

private:
    Error code_;
    std::string message_;

    explicit Status(Error code) noexcept : code_(code) {}
    Status(Error code, std::string message) : code_{code}, message_{std::move(message)} {}
    static const char* ErrorName(Error code)
    {
        switch (code) {
#define X(name, init) \
    case Error::name: return #name;
            UCM_STATUS_ERROR_LIST
#undef X
        }
        return "Unknown";
    }

public:
    bool operator==(const Status& other) const noexcept { return code_ == other.code_; }
    bool operator!=(const Status& other) const noexcept { return !(*this == other); }
    [[nodiscard]] int32_t Underlying() const noexcept { return static_cast<int32_t>(code_); }
    [[nodiscard]] std::string ToString() const
    {
        if (message_.empty()) {
            return fmt::format("{} ({})", ErrorName(code_), static_cast<int32_t>(code_));
        }
        return fmt::format("{} ({}):: {}", ErrorName(code_), static_cast<int32_t>(code_), message_);
    }
    constexpr bool Success() const noexcept { return code_ == Error::Ok; }
    constexpr bool Failure() const noexcept { return !Success(); }

#define X(name, init)                                                                 \
    [[nodiscard]] static Status name() noexcept { return Status{Error::name}; }       \
    template <typename... Args>                                                       \
    [[nodiscard]] static Status name(fmt::format_string<Args...> fmt, Args&&... args) \
    {                                                                                 \
        return Make(Error::name, fmt, std::forward<Args>(args)...);                   \
    }
    UCM_STATUS_ERROR_LIST
#undef X

    [[nodiscard]] static Status Make(Error code) noexcept { return Status{code}; }
    template <typename... Args>
    [[nodiscard]] static Status Make(Error code, fmt::format_string<Args...> fmt, Args&&... args)
    {
        return {code, fmt::format(fmt, std::forward<Args>(args)...)};
    }
};

#undef UCM_STATUS_ERROR_LIST

template <class T>
class Expected {
    std::variant<Status, T> v_;

public:
    template <typename U = T, std::enable_if_t<!std::is_same_v<std::decay_t<U>, Expected> &&
                                                   !std::is_same_v<std::decay_t<U>, Status>,
                                               int> = 0>
    Expected(U&& val) : v_(std::forward<U>(val))
    {
    }
    Expected(Status err) noexcept : v_(err) {}

    bool HasValue() const noexcept { return v_.index() == 1; }
    explicit operator bool() const noexcept { return HasValue(); }

    T& Value() & { return std::get<T>(v_); }
    const T& Value() const& { return std::get<T>(v_); }
    T&& Value() && { return std::get<T>(std::move(v_)); }

    T& operator*() & { return std::get<T>(v_); }
    const T& operator*() const& { return std::get<T>(v_); }
    T&& operator*() && { return std::get<T>(std::move(v_)); }

    T* operator->() { return &std::get<T>(v_); }
    const T* operator->() const { return &std::get<T>(v_); }

    const Status& Error() const& { return std::get<Status>(v_); }
    Status& Error() & { return std::get<Status>(v_); }
    Status&& Error() && { return std::get<Status>(std::move(v_)); }

    template <typename U>
    T ValueOr(U&& defaultValue) const&
    {
        return HasValue() ? Value() : static_cast<T>(std::forward<U>(defaultValue));
    }
    template <typename U>
    T ValueOr(U&& defaultValue) &&
    {
        return HasValue() ? std::move(Value()) : static_cast<T>(std::forward<U>(defaultValue));
    }
};

inline std::string format_as(const Status& status) { return status.ToString(); }

}  // namespace UC
