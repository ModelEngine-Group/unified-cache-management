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
#include <string>
#include <unordered_map>
#include "utils/config_parse.h"

/**
 * 日志配置模块：从配置文件提取 log.* 键并做类型转换，与日志库解耦——
 * 级别名转枚举与文件名时间标记展开由 LoggerManager 处理；getter 缺 key /
 * 坏值抛异常（由 main 统一 catch），Parse 保持 bool + err 契约。
 */
class LogConfigUtil {
public:
    bool Parse(const std::unordered_map<std::string, std::string> &items, std::string &err);

    std::string GetDir() const { return config_parse::RequireValue(values_, "log.dir"); }
    std::string GetLevel() const { return config_parse::RequireValue(values_, "log.level"); }
    bool GetConsole() const { return config_parse::RequireValue(values_, "log.console") == "on"; }
    std::string GetPattern() const { return config_parse::RequireValue(values_, "log.pattern"); }
    std::string GetFileName() const { return config_parse::RequireValue(values_, "log.file_name"); }
    std::string GetGetDir() const { return config_parse::RequireValue(values_, "log.get_dir"); }
    std::string GetGetFileName() const
    {
        return config_parse::RequireValue(values_, "log.get_file_name");
    }
    std::string GetOtherDir() const { return config_parse::RequireValue(values_, "log.other_dir"); }
    std::string GetOtherFileName() const
    {
        return config_parse::RequireValue(values_, "log.other_file_name");
    }
    std::size_t GetMaxFileSize() const
    {
        return config_parse::RequireSize(values_, "log.max_file_size");
    }
    std::size_t GetMaxFiles() const { return config_parse::RequireSize(values_, "log.max_files"); }
    std::size_t GetAsyncQueueSize() const
    {
        return config_parse::RequireSize(values_, "log.async_queue_size");
    }
    int GetAsyncFlushIntervalMs() const
    {
        return config_parse::RequireInt(values_, "log.async_flush_interval_ms");
    }

    void Print() const;

private:
    std::unordered_map<std::string, std::string> values_;
};
