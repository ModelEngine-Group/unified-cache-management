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

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

/**
 * 配置键值解析工具（namespace 形式，ConfigUtil 与 LogConfigUtil 共用）：
 * 缺 key / 坏值抛 std::runtime_error，由调用方统一 catch。
 */
namespace config_parse {

/** 取 key 对应字符串值；缺失抛异常 */
std::string RequireValue(const std::unordered_map<std::string, std::string> &values,
                         const std::string &key);

/** 解析字符串为 int；坏值抛异常 */
int ParseInt(const std::string &key, const std::string &value);

/** 解析字符串为 int64；坏值抛异常 */
std::int64_t ParseInt64(const std::string &key, const std::string &value);

/** 解析字符串为非负 size_t；负数或坏值抛异常 */
std::size_t ParseSize(const std::string &key, const std::string &value);

/** 取 key 对应 int 值；缺失或坏值抛异常 */
int RequireInt(const std::unordered_map<std::string, std::string> &values, const std::string &key);

/** 取 key 对应 int 值；key 缺失返回 fallback（新增可选键兼容存量配置），坏值抛异常 */
int OptionalInt(const std::unordered_map<std::string, std::string> &values, const std::string &key,
                int fallback);

/** 取 key 对应字符串值；key 缺失返回 fallback（新增可选键兼容存量配置） */
std::string OptionalValue(const std::unordered_map<std::string, std::string> &values,
                          const std::string &key, const std::string &fallback);

/** 取 key 对应 int64 值；缺失或坏值抛异常 */
std::int64_t RequireInt64(const std::unordered_map<std::string, std::string> &values,
                          const std::string &key);

/** 取 key 对应非负 size_t；缺失、负数或坏值抛异常 */
std::size_t RequireSize(const std::unordered_map<std::string, std::string> &values,
                        const std::string &key);

}  // namespace config_parse
