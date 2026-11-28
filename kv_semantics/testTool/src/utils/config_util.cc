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

#include "utils/config_util.h"
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "log/logger_manager.h"
#include "utils/config_parse.h"
#include "utils/file_util.h"

namespace {

/** 去除首尾空白 */
std::string Trim(const std::string &s)
{
    const char *kWhitespace = " \t\r\n\v\f";
    size_t begin = s.find_first_not_of(kWhitespace);
    if (begin == std::string::npos) { return std::string(); }
    size_t end = s.find_last_not_of(kWhitespace);
    return s.substr(begin, end - begin + 1);
}

}  // namespace

ConfigUtil &ConfigUtil::GetInstance()
{
    static ConfigUtil instance;
    return instance;
}

bool ConfigUtil::Load(const std::string &path, std::string &err)
{
    std::vector<std::string> lines;
    if (!FileUtil::ReadLines(path, lines, err)) { return false; }

    // 逐行解析为键值表；'#' 起整行或行内注释作用；log.* 前缀键分流给日志配置模块
    std::unordered_map<std::string, std::string> values;
    std::unordered_map<std::string, std::string> logItems;
    for (size_t lineNo = 1; lineNo <= lines.size(); ++lineNo) {
        const std::string trimmed = Trim(lines[lineNo - 1]);
        if (trimmed.empty() || trimmed[0] == '#') { continue; }
        size_t sep = trimmed.find('=');
        if (sep == std::string::npos) {
            err = "config line " + std::to_string(lineNo) + " missing '=': " + trimmed;
            return false;
        }
        std::string key = Trim(trimmed.substr(0, sep));
        std::string value = Trim(trimmed.substr(sep + 1));
        // 行内注释：仅 '#' 前为空白时视为注释起点（值本身可含 '#'，如 log.pattern 的 %s:%#）
        size_t hash = std::string::npos;
        for (size_t i = 0; i < value.size(); ++i) {
            if (value[i] == '#' && (i == 0 || value[i - 1] == ' ' || value[i - 1] == '\t')) {
                hash = i;
                break;
            }
        }
        if (hash != std::string::npos) { value = Trim(value.substr(0, hash)); }
        if (key.empty()) {
            err = "config line " + std::to_string(lineNo) + " has empty key";
            return false;
        }
        auto &dst = (key.rfind("log.", 0) == 0) ? logItems : values;
        if (!dst.emplace(key, value).second) {
            err = "config line " + std::to_string(lineNo) + " has duplicate key: " + key;
            return false;
        }
    }
    values_ = std::move(values);

    // 日志项经 LogConfigUtil 解析
    if (!logConfig_.Parse(logItems, err)) { return false; }

    // Check 的异常转为 err，保持 bool 契约
    try {
        if (!Check(err)) { return false; }
    } catch (const std::exception &e) {
        err = e.what();
        return false;
    }
    return true;
}

bool ConfigUtil::Check(std::string &err)
{
    // runtime=0（循环模式）时 loop 必须大于 0
    const int rt = config_parse::RequireInt(values_, "runtime");
    const int lp = config_parse::RequireInt(values_, "loop");
    if (rt == 0 && lp == 0) {
        err = "loop must be > 0 when runtime=0 (loop mode)";
        return false;
    }

    // 多线程参数：至少 1 个 worker；起始绑核编号非负（worker i 绑 startcoreid+i）
    const int numJobs = config_parse::RequireInt(values_, "numjobs");
    const int startCoreId = config_parse::RequireInt(values_, "startcoreid");
    if (numJobs <= 0 || startCoreId < 0) {
        err = "invalid worker config: numjobs=" + std::to_string(numJobs) +
              " startcoreid=" + std::to_string(startCoreId);
        return false;
    }

    // 值缓冲池参数必须非零（负值已由 RequireSize 拦截）
    const std::size_t bufferSize = config_parse::RequireSize(values_, "buffersize");
    const std::size_t bufferCount = config_parse::RequireSize(values_, "buffercount");
    if (bufferSize == 0 || bufferCount == 0) {
        err = "invalid value buffer pool config: buffersize=" + std::to_string(bufferSize) +
              " buffercount=" + std::to_string(bufferCount);
        return false;
    }

    // autodisconnect 负值拦截
    const int autoDisconnect = GetAutoDisconnect();
    if (autoDisconnect < 0) {
        err = "invalid autodisconnect: " + std::to_string(autoDisconnect) +
              " (expected seconds >= 0, 0 means disabled)";
        return false;
    }

    // ratelimit_mb 负值拦截（0 不限速）
    const int rateLimitMb = GetRateLimitMb();
    if (rateLimitMb < 0) {
        err = "invalid ratelimit_mb: " + std::to_string(rateLimitMb) +
              " (expected MB/s >= 0, 0 means unlimited)";
        return false;
    }

    // p99_interval_s 负值拦截（0 禁用打印）
    const int p99Interval = GetP99IntervalSecs();
    if (p99Interval < 0) {
        err = "invalid p99_interval_s: " + std::to_string(p99Interval) +
              " (expected seconds >= 0, 0 means disabled)";
        return false;
    }

    // metrics_exporter_port 合法性拦截（1-65535；exporter 开启与否都先校验值域）
    const int exporterPort = GetMetricsExporterPort();
    if (exporterPort <= 0 || exporterPort > 65535) {
        err = "invalid metrics_exporter_port: " + std::to_string(exporterPort) +
              " (expected 1-65535)";
        return false;
    }
    return true;
}

void ConfigUtil::Print() const
{
    LOG_INFO("testmode={}", GetTestMode());
    LOG_INFO("runtime={}", GetRuntime());
    LOG_INFO("loop={}", GetLoop());
    LOG_INFO("perf={}", GetPerf());
    LOG_INFO("onlyconnect={}", GetOnlyConnect());
    LOG_INFO("autodisconnect={}", GetAutoDisconnect());
    LOG_INFO("ratelimit_mb={}", GetRateLimitMb());
    LOG_INFO("p99_interval_s={}", GetP99IntervalSecs());
    LOG_INFO("metrics_exporter={}", GetMetricsExporter() ? "on" : "off");
    LOG_INFO("metrics_exporter_port={}", GetMetricsExporterPort());
    LOG_INFO("startcoreid={}", GetStartCoreId());
    LOG_INFO("numjobs={}", GetNumJobs());
    LOG_INFO("buffersize={}", GetBufferSize());
    LOG_INFO("buffercount={}", GetBufferCount());
    LOG_INFO("buffertype={}", GetBufferType());
    LOG_INFO("deviceid={}", GetDeviceId());
    logConfig_.Print();
}
