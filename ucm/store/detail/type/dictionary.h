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
#ifndef UNIFIEDCACHE_STORE_DETAIL_TYPE_DICTIONARY_H
#define UNIFIEDCACHE_STORE_DETAIL_TYPE_DICTIONARY_H

#include <any>
#include <string>
#include <unordered_map>

namespace UC::Detail {

class Dictionary {
    std::unordered_map<std::string, std::any> data_;

    template <typename T>
    T Get(const std::string& key) const
    {
        return std::any_cast<T>(data_.find(key)->second);
    }

public:
    bool Contains(const std::string& key) const { return data_.find(key) != data_.end(); }
    template <typename T>
    void Set(const std::string& key, const T& value)
    {
        data_[key] = value;
    }
    template <typename T>
    void SetNumber(const std::string& key, const T& value)
    {
        data_[key] = static_cast<ssize_t>(value);
    }
    template <typename T>
    void Get(const std::string& key, T& target) const
    {
        if (Contains(key)) { target = Get<T>(key); }
    }
    template <typename T>
    void GetNumber(const std::string& key, T& target) const
    {
        if (Contains(key)) { target = static_cast<T>(Get<ssize_t>(key)); }
    }
};

}  // namespace UC::Detail

#endif  // UNIFIEDCACHE_STORE_DETAIL_TYPE_DICTIONARY_H
