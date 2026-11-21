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
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "log/logger_manager.h"
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

/** 级别名转枚举（小写标准名），成功写入 out */
bool StringToLevel(const std::string &name, spdlog::level::level_enum &out)
{
    if (name == "trace") {
        out = spdlog::level::trace;
    } else if (name == "debug") {
        out = spdlog::level::debug;
    } else if (name == "info") {
        out = spdlog::level::info;
    } else if (name == "warn") {
        out = spdlog::level::warn;
    } else if (name == "error") {
        out = spdlog::level::err;
    } else if (name == "critical") {
        out = spdlog::level::critical;
    } else {
        return false;
    }
    return true;
}

/** 按给定时刻展开文件名模板中的 strftime 时间标记（%Y%m%d_%H 等）；仅初始化阶段单线程调用 */
std::string ExpandFileNameTpl(const std::string &tpl, std::time_t t)
{
    std::tm *tmNow = std::localtime(&t);
    if (tmNow == nullptr) { return tpl; }
    std::ostringstream oss;
    oss << std::put_time(tmNow, tpl.c_str());
    return oss.str();
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

    // 逐行解析为原始键值表；'#' 起整行或行内注释作用
    std::unordered_map<std::string, std::string> items;
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
        size_t hash = value.find('#');
        if (hash != std::string::npos) { value = Trim(value.substr(0, hash)); }
        if (key.empty()) {
            err = "config line " + std::to_string(lineNo) + " has empty key";
            return false;
        }
        if (!items.emplace(key, value).second) {
            err = "config line " + std::to_string(lineNo) + " has duplicate key: " + key;
            return false;
        }
    }
    values_ = std::move(items);

    // Check 与文件名展开走安全访问，缺失/坏值异常在此转为 err，保持 Load 的 bool 契约
    try {
        if (!Check(err)) { return false; }

        // 文件名模板按加载时刻展开时间标记，并防止展开后重名
        std::time_t now = std::time(nullptr);
        std::string fullName = ExpandFileNameTpl(RequireValue("log.file_name"), now);
        std::string warnName = ExpandFileNameTpl(RequireValue("log.warn_file_name"), now);
        std::string errorName = ExpandFileNameTpl(RequireValue("log.error_file_name"), now);
        if (fullName == warnName || fullName == errorName || warnName == errorName) {
            err =
                "expanded file names of log.file_name / log.warn_file_name / log.error_file_name "
                "conflict";
            return false;
        }
        values_["log.file_name"] = std::move(fullName);
        values_["log.warn_file_name"] = std::move(warnName);
        values_["log.error_file_name"] = std::move(errorName);
    } catch (const std::exception &e) {
        err = e.what();
        return false;
    }
    return true;
}

bool ConfigUtil::Check(std::string &err)
{
    // 级别名合法
    spdlog::level::level_enum level;
    if (!StringToLevel(RequireValue("log.level"), level)) {
        err = "invalid log.level: " + RequireValue("log.level") +
              " (expected trace/debug/info/warn/error/critical)";
        return false;
    }

    // runtime=0（循环模式）时 loop 必须大于 0
    const int rt = RequireInt("runtime");
    const int lp = RequireInt("loop");
    if (rt == 0 && lp == 0) {
        err = "loop must be > 0 when runtime=0 (loop mode)";
        return false;
    }

    // 多线程参数：至少 1 个 worker；起始绑核编号非负（worker i 绑 startcoreid+i）
    const int numJobs = RequireInt("numjobs");
    const int startCoreId = RequireInt("startcoreid");
    if (numJobs <= 0 || startCoreId < 0) {
        err = "invalid worker config: numjobs=" + std::to_string(numJobs) +
              " startcoreid=" + std::to_string(startCoreId);
        return false;
    }

    // 值缓冲池参数必须为正：非正值会在后续分配/注册中产生空区域
    // （buffersize 为字节量留 int64，buffercount 为块数用 int，与 getter 值域一致）
    const int64_t bufferSize = RequireInt64("buffersize");
    const int bufferCount = RequireInt("buffercount");
    if (bufferSize <= 0 || bufferCount <= 0) {
        err = "invalid value buffer pool config: buffersize=" + std::to_string(bufferSize) +
              " buffercount=" + std::to_string(bufferCount);
        return false;
    }

    // log.console 可选键：存在时值须为 on / off（缺省视为 on，见 GetLogConsole）
    const auto consoleIt = values_.find("log.console");
    if (consoleIt != values_.end() && consoleIt->second != "on" && consoleIt->second != "off") {
        err = "invalid log.console: " + consoleIt->second + " (expected on or off)";
        return false;
    }

    // autodisconnect 可选键：负值无意义（0 缺省禁用、>0 生效），前置拦截避免静默误配
    const int autoDisconnect = GetAutoDisconnect();
    if (autoDisconnect < 0) {
        err = "invalid autodisconnect: " + std::to_string(autoDisconnect) +
              " (expected seconds >= 0, 0 means disabled)";
        return false;
    }
    return true;
}

std::string ConfigUtil::RequireValue(const std::string &key) const
{
    const auto it = values_.find(key);
    if (it == values_.end()) { throw std::runtime_error("config key missing: " + key); }
    return it->second;
}

int ConfigUtil::RequireInt(const std::string &key) const
{
    const std::string value = RequireValue(key);
    try {
        return std::stoi(value);
    } catch (const std::exception &) {
        throw std::runtime_error("config key '" + key + "' invalid int value: '" + value + "'");
    }
}

std::int64_t ConfigUtil::RequireInt64(const std::string &key) const
{
    const std::string value = RequireValue(key);
    try {
        return std::stoll(value);
    } catch (const std::exception &) {
        throw std::runtime_error("config key '" + key + "' invalid int64 value: '" + value + "'");
    }
}

int ConfigUtil::GetTestMode() const { return RequireInt("testmode"); }

int ConfigUtil::GetRuntime() const { return RequireInt("runtime"); }

int ConfigUtil::GetLoop() const { return RequireInt("loop"); }

bool ConfigUtil::GetPerf() const { return RequireValue("perf") == "on"; }

bool ConfigUtil::GetOnlyConnect() const { return RequireValue("onlyconnect") == "on"; }

int ConfigUtil::GetAutoDisconnect() const
{
    // 可选键：缺省 0（禁用，老配置无需改动）；存在时按整数秒解析，坏值与必填键同款抛异常可感知
    const auto it = values_.find("autodisconnect");
    if (it == values_.end()) { return 0; }
    try {
        return std::stoi(it->second);
    } catch (const std::exception &) {
        throw std::runtime_error("config key 'autodisconnect' invalid int value: '" + it->second +
                                 "'");
    }
}

int ConfigUtil::GetStartCoreId() const { return RequireInt("startcoreid"); }

int ConfigUtil::GetNumJobs() const { return RequireInt("numjobs"); }

int64_t ConfigUtil::GetBufferSize() const { return RequireInt64("buffersize"); }

int ConfigUtil::GetBufferCount() const { return RequireInt("buffercount"); }

std::string ConfigUtil::GetBufferType() const { return RequireValue("buffertype"); }
int ConfigUtil::GetDeviceId() const { return RequireInt("deviceid"); }

std::string ConfigUtil::GetReadLogDir() const { return RequireValue("readlogdir"); }

std::string ConfigUtil::GetLogDir() const { return RequireValue("logdir"); }

spdlog::level::level_enum ConfigUtil::GetLogLevel() const
{
    spdlog::level::level_enum level = spdlog::level::info;
    (void)StringToLevel(RequireValue("log.level"), level);
    return level;
}

bool ConfigUtil::GetLogConsole() const
{
    // 可选键：缺省视为 on（默认打印到控制台）；存在时仅 "on" 打开（与 perf 的宽松解析同款，
    // 非法值已在 Check 前置拦截）
    const auto it = values_.find("log.console");
    return it == values_.end() || it->second == "on";
}

std::string ConfigUtil::GetLogPattern() const { return RequireValue("log.pattern"); }

std::string ConfigUtil::GetLogFileName() const { return RequireValue("log.file_name"); }

std::string ConfigUtil::GetLogWarnFileName() const { return RequireValue("log.warn_file_name"); }

std::string ConfigUtil::GetLogErrorFileName() const { return RequireValue("log.error_file_name"); }

int64_t ConfigUtil::GetLogMaxFileSize() const { return RequireInt64("log.max_file_size"); }

int ConfigUtil::GetLogMaxFiles() const { return RequireInt("log.max_files"); }

int ConfigUtil::GetLogAsyncQueueSize() const { return RequireInt("log.async_queue_size"); }

int ConfigUtil::GetLogAsyncFlushIntervalMs() const
{
    return RequireInt("log.async_flush_interval_ms");
}

void ConfigUtil::Print() const
{
    LOG_INFO("testmode={}", GetTestMode());
    LOG_INFO("runtime={}", GetRuntime());
    LOG_INFO("loop={}", GetLoop());
    LOG_INFO("perf={}", GetPerf());
    LOG_INFO("onlyconnect={}", GetOnlyConnect());
    LOG_INFO("autodisconnect={}", GetAutoDisconnect());
    LOG_INFO("startcoreid={}", GetStartCoreId());
    LOG_INFO("numjobs={}", GetNumJobs());
    LOG_INFO("buffersize={}", GetBufferSize());
    LOG_INFO("buffercount={}", GetBufferCount());
    LOG_INFO("buffertype={}", GetBufferType());
    LOG_INFO("deviceid={}", GetDeviceId());
    LOG_INFO("readlogdir={}", GetReadLogDir());
    LOG_INFO("logdir={}", GetLogDir());

    LOG_INFO("log.level={}", spdlog::level::to_string_view(GetLogLevel()));
    LOG_INFO("log.console={}", GetLogConsole());
    LOG_INFO("log.pattern={}", GetLogPattern());
    LOG_INFO("log.file_name={}", GetLogFileName());
    LOG_INFO("log.warn_file_name={}", GetLogWarnFileName());
    LOG_INFO("log.error_file_name={}", GetLogErrorFileName());
    LOG_INFO("log.max_file_size={}", GetLogMaxFileSize());
    LOG_INFO("log.max_files={}", GetLogMaxFiles());
    LOG_INFO("log.async_queue_size={}", GetLogAsyncQueueSize());
    LOG_INFO("log.async_flush_interval_ms={}", GetLogAsyncFlushIntervalMs());
}
