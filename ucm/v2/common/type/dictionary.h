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

#include <algorithm>
#include <any>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "status/status.h"

namespace UC {

/**
 * @brief Key-value store backed by std::any for runtime type erasure.
 */
class Dictionary {
    std::unordered_map<std::string, std::any> data_;

public:
    /** Check whether a key exists in the dictionary. */
    bool Contains(const std::string& key) const { return data_.count(key) != 0; }
    /**
     * Store a value of type T under the given key.
     * Array types are rejected to prevent silent pointer decay.
     * */
    template <typename T, typename = std::enable_if_t<!std::is_array_v<std::remove_reference_t<T>>>>
    void Set(const std::string& key, T&& value)
    {
        data_[key] = std::forward<T>(value);
    }
    /** Store a string literal (or char pointer) under the given key as std::string. */
    void Set(const std::string& key, const char* value) { data_[key] = std::string(value); }
    /** Store an already type-erased value under the given key. */
    void SetAny(const std::string& key, const std::any& value) { data_[key] = value; }
    /** Retrieve a value of type T; returns false on absent key or type mismatch. */
    template <typename T>
    [[nodiscard]] bool Get(const std::string& key, T& target) const
    {
        auto it = data_.find(key);
        if (it == data_.end()) { return false; }
        if (const auto* value = std::any_cast<T>(&it->second)) {
            target = *value;
            return true;
        }
        return false;
    }
    /**
     * Visit every stored key-value pair, in unspecified order.
     *
     * Read-only traversal for system layers that merge or validate whole
     * dictionaries; typed access stays with Get()/GetNumbers().
     * */
    template <typename Visitor>
    void ForEach(Visitor&& visitor) const
    {
        for (const auto& item : data_) { visitor(item.first, item.second); }
    }
    /** Store a std::int64_t value under the given key. */
    void SetInt64(const std::string& key, std::int64_t value) { data_[key] = value; }
    /** Retrieve a std::int64_t value; returns false on absent key or type mismatch. */
    [[nodiscard]] bool GetInt64(const std::string& key, std::int64_t& target) const
    {
        return Get(key, target);
    }
    /** Store a double value under the given key. */
    void SetDouble(const std::string& key, double value) { data_[key] = value; }
    /** Retrieve a double value; returns false on absent key or type mismatch. */
    [[nodiscard]] bool GetDouble(const std::string& key, double& target) const
    {
        return Get(key, target);
    }
    /** Store a numeric vector as std::int64_t elements under the given key. */
    template <typename T>
    void SetNumbers(const std::string& key, const std::vector<T>& values)
    {
        data_[key] = std::vector<std::int64_t>(values.begin(), values.end());
    }
    /**
     * Retrieve a numeric vector cast element-wise to T, replacing the contents of
     * target; returns false on absent key or type mismatch.
     * */
    template <typename T>
    [[nodiscard]] bool GetNumbers(const std::string& key, std::vector<T>& target) const
    {
        auto it = data_.find(key);
        if (it == data_.end()) { return false; }
        if (const auto* values = std::any_cast<std::vector<std::int64_t>>(&it->second)) {
            target.clear();
            target.reserve(values->size());
            for (const auto& d : *values) { target.push_back(static_cast<T>(d)); }
            return true;
        }
        return false;
    }
};

/**
 * @brief Sequential field reader over a Dictionary with fail-fast validation.
 *
 * Each read validates one key; after the first failure every later read is a
 * no-op and Result() keeps the first error. Optional reads leave the target
 * untouched when the key is absent and reject present-but-invalid values.
 */
class DictionaryReader {
    const Dictionary& dict_;
    std::string context_;
    Status status_{Status::Ok()};

public:
    explicit DictionaryReader(const Dictionary& dict, std::string context = "")
        : dict_{dict}, context_{std::move(context)}
    {
    }

    /** Result of all reads: Ok() until one fails, then the first failure. */
    [[nodiscard]] const Status& Result() const noexcept { return status_; }

    /** Read a required string; the key must be present and the value non-empty. */
    void RequiredNonEmpty(const char* key, std::string& target)
    {
        if (status_.Failure()) { return; }
        if (!dict_.Get(key, target) || target.empty()) {
            Fail("'{}' is required and must be non-empty", key);
        }
    }

    /** Read an optional string; absent keeps the target, present must be non-empty. */
    void OptionalNonEmpty(const char* key, std::string& target)
    {
        if (status_.Failure()) { return; }
        if (!dict_.Contains(key)) { return; }
        if (!dict_.Get(key, target) || target.empty()) {
            Fail("'{}' must be a non-empty string when set", key);
        }
    }

    /** Read a required string restricted to the listed values. */
    void RequiredEnum(const char* key, std::string& target,
                      std::initializer_list<std::string_view> values)
    {
        if (status_.Failure()) { return; }
        if (!dict_.Get(key, target) ||
            std::find(values.begin(), values.end(), std::string_view{target}) == values.end()) {
            Fail("'{}' must be one of {}, got '{}'", key, Join(values), target);
        }
    }

    /** Read a required strictly positive integer of integral type T. */
    template <typename T>
    void RequiredPositive(const char* key, T& target)
    {
        if (status_.Failure()) { return; }
        std::int64_t value = 0;
        if (!dict_.GetInt64(key, value) || value <= 0 || !FitsIn<T>(value)) {
            Fail("'{}' must be a positive integer", key);
            return;
        }
        target = static_cast<T>(value);
    }

    /** Read an optional strictly positive integer of integral type T. */
    template <typename T>
    void OptionalPositive(const char* key, T& target)
    {
        if (status_.Failure()) { return; }
        if (!dict_.Contains(key)) { return; }
        std::int64_t value = 0;
        if (!dict_.GetInt64(key, value) || value <= 0 || !FitsIn<T>(value)) {
            Fail("'{}' must be a positive integer", key);
            return;
        }
        target = static_cast<T>(value);
    }

    /** Read a required integer that must fit the int32 range (negatives allowed). */
    void RequiredInt32(const char* key, std::int32_t& target)
    {
        if (status_.Failure()) { return; }
        std::int64_t value = 0;
        if (!dict_.GetInt64(key, value) ||
            value < static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()) ||
            value > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
            Fail("'{}' must be an integer in int32 range", key);
            return;
        }
        target = static_cast<std::int32_t>(value);
    }

    /** Read an optional boolean; absent keeps the target, present must be a bool. */
    void OptionalBool(const char* key, bool& target)
    {
        if (status_.Failure()) { return; }
        if (!dict_.Contains(key)) { return; }
        if (!dict_.Get(key, target)) { Fail("'{}' must be a bool when set", key); }
    }

private:
    /** Record the first failure, prefixed with the reader's context. */
    template <typename... Args>
    void Fail(fmt::format_string<Args...> spec, Args&&... args)
    {
        if (status_.Failure()) { return; }
        std::string message = fmt::format(spec, std::forward<Args>(args)...);
        status_ = context_.empty() ? Status::InvalidParam("{}", message)
                                   : Status::InvalidParam("{}: {}", context_, message);
    }

    /** Whether a positive int64 value fits the integral type T. */
    template <typename T>
    static bool FitsIn(std::int64_t value) noexcept
    {
        static_assert(std::is_integral_v<T>, "T must be an integral type");
        if constexpr (std::is_unsigned_v<T>) {
            if constexpr (sizeof(T) >= sizeof(std::int64_t)) {
                return true;
            } else {
                return value <= static_cast<std::int64_t>(std::numeric_limits<T>::max());
            }
        } else if constexpr (sizeof(T) >= sizeof(std::int64_t)) {
            return true;
        } else {
            return value >= static_cast<std::int64_t>(std::numeric_limits<T>::min()) &&
                   value <= static_cast<std::int64_t>(std::numeric_limits<T>::max());
        }
    }

    /** Join the enum values with '/' for error messages. */
    static std::string Join(std::initializer_list<std::string_view> values)
    {
        std::string joined;
        for (const auto& value : values) {
            if (!joined.empty()) { joined += '/'; }
            joined.append(value);
        }
        return joined;
    }
};

}  // namespace UC
