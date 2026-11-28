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

#include "utils/log_config_util.h"
#include "log/logger_manager.h"

bool LogConfigUtil::Parse(const std::unordered_map<std::string, std::string> &items,
                          std::string &err)
{
    values_ = items;

    // log.console 值须为 on / off（缺键报错，坏值拦截）
    const auto consoleIt = values_.find("log.console");
    if (consoleIt == values_.end()) {
        err = "config key missing: log.console";
        return false;
    }
    if (consoleIt->second != "on" && consoleIt->second != "off") {
        err = "invalid log.console: " + consoleIt->second + " (expected on or off)";
        return false;
    }
    return true;
}

void LogConfigUtil::Print() const
{
    LOG_INFO("log.dir={}", GetDir());
    LOG_INFO("log.level={}", GetLevel());
    LOG_INFO("log.console={}", GetConsole());
    LOG_INFO("log.pattern={}", GetPattern());
    LOG_INFO("log.file_name={}", GetFileName());
    LOG_INFO("log.get_dir={}", GetGetDir());
    LOG_INFO("log.get_file_name={}", GetGetFileName());
    LOG_INFO("log.other_dir={}", GetOtherDir());
    LOG_INFO("log.other_file_name={}", GetOtherFileName());
    LOG_INFO("log.max_file_size={}", GetMaxFileSize());
    LOG_INFO("log.max_files={}", GetMaxFiles());
    LOG_INFO("log.async_queue_size={}", GetAsyncQueueSize());
    LOG_INFO("log.async_flush_interval_ms={}", GetAsyncFlushIntervalMs());
}
