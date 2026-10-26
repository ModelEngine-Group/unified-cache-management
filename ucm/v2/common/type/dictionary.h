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

#include <any>
#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

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

}  // namespace UC
