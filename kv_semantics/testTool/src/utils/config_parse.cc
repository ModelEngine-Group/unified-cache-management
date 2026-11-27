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

#include "utils/config_parse.h"
#include <stdexcept>

namespace config_parse {

std::string RequireValue(const std::unordered_map<std::string, std::string> &values,
                         const std::string &key)
{
    const auto it = values.find(key);
    if (it == values.end()) { throw std::runtime_error("config key missing: " + key); }
    return it->second;
}

int ParseInt(const std::string &key, const std::string &value)
{
    try {
        return std::stoi(value);
    } catch (const std::exception &) {
        throw std::runtime_error("config key '" + key + "' invalid int value: '" + value + "'");
    }
}

std::int64_t ParseInt64(const std::string &key, const std::string &value)
{
    try {
        return std::stoll(value);
    } catch (const std::exception &) {
        throw std::runtime_error("config key '" + key + "' invalid int64 value: '" + value + "'");
    }
}

std::size_t ParseSize(const std::string &key, const std::string &value)
{
    try {
        const std::int64_t parsed = std::stoll(value);
        if (parsed < 0) { throw std::invalid_argument("negative"); }
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception &) {
        throw std::runtime_error("config key '" + key + "' invalid size value: '" + value + "'");
    }
}

int RequireInt(const std::unordered_map<std::string, std::string> &values, const std::string &key)
{
    return ParseInt(key, RequireValue(values, key));
}

std::int64_t RequireInt64(const std::unordered_map<std::string, std::string> &values,
                          const std::string &key)
{
    return ParseInt64(key, RequireValue(values, key));
}

std::size_t RequireSize(const std::unordered_map<std::string, std::string> &values,
                        const std::string &key)
{
    return ParseSize(key, RequireValue(values, key));
}

}  // namespace config_parse
