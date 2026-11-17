/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "utils/config_util.h"

#include <cstdint>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <vector>

#include "log/logger_manager.h"
#include "utils/file_util.h"

namespace {

/** 去除首尾空白 */
std::string Trim(const std::string &s)
{
    const char *kWhitespace = " \t\r\n\v\f";
    size_t begin = s.find_first_not_of(kWhitespace);
    if (begin == std::string::npos) {
        return std::string();
    }
    size_t end = s.find_last_not_of(kWhitespace);
    return s.substr(begin, end - begin + 1);
}

/** 级别名转枚举（小写标准名），成功写入 out */
bool StringToLevel(const std::string &name, spdlog::level::level_enum &out)
{
    if (name == "trace") { out = spdlog::level::trace; }
    else if (name == "debug") { out = spdlog::level::debug; }
    else if (name == "info") { out = spdlog::level::info; }
    else if (name == "warn") { out = spdlog::level::warn; }
    else if (name == "error") { out = spdlog::level::err; }
    else if (name == "critical") { out = spdlog::level::critical; }
    else { return false; }
    return true;
}

/** 按给定时刻展开文件名模板中的 strftime 时间标记（%Y%m%d_%H 等）；仅初始化阶段单线程调用 */
std::string ExpandFileNameTpl(const std::string &tpl, std::time_t t)
{
    std::tm *tmNow = std::localtime(&t);
    if (tmNow == nullptr) {
        return tpl;
    }
    std::ostringstream oss;
    oss << std::put_time(tmNow, tpl.c_str());
    return oss.str();
}

} // namespace

ConfigUtil &ConfigUtil::GetInstance()
{
    static ConfigUtil instance;
    return instance;
}

bool ConfigUtil::Load(const std::string &path, std::string &err)
{
    std::vector<std::string> lines;
    if (!FileUtil::ReadLines(path, lines, err)) {
        return false;
    }

    // 逐行解析为原始键值表；'#' 起整行或行内注释作用
    std::unordered_map<std::string, std::string> items;
    for (size_t lineNo = 1; lineNo <= lines.size(); ++lineNo) {
        const std::string trimmed = Trim(lines[lineNo - 1]);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        size_t sep = trimmed.find('=');
        if (sep == std::string::npos) {
            err = "config line " + std::to_string(lineNo) + " missing '=': " + trimmed;
            return false;
        }
        std::string key = Trim(trimmed.substr(0, sep));
        std::string value = Trim(trimmed.substr(sep + 1));
        size_t hash = value.find('#');
        if (hash != std::string::npos) {
            value = Trim(value.substr(0, hash));
        }
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

    if (!Check(err)) {
        return false;
    }

    // 文件名模板按加载时刻展开时间标记，并防止展开后重名
    std::time_t now = std::time(nullptr);
    std::string fullName = ExpandFileNameTpl(values_.at("log.file_name"), now);
    std::string warnName = ExpandFileNameTpl(values_.at("log.warn_file_name"), now);
    std::string errorName = ExpandFileNameTpl(values_.at("log.error_file_name"), now);
    if (fullName == warnName || fullName == errorName || warnName == errorName) {
        err = "expanded file names of log.file_name / log.warn_file_name / log.error_file_name conflict";
        return false;
    }
    values_["log.file_name"] = std::move(fullName);
    values_["log.warn_file_name"] = std::move(warnName);
    values_["log.error_file_name"] = std::move(errorName);
    return true;
}

bool ConfigUtil::Check(std::string &err)
{
    // 级别名合法
    spdlog::level::level_enum level;
    if (!StringToLevel(values_.at("log.level"), level)) {
        err = "invalid log.level: " + values_.at("log.level") +
              " (expected trace/debug/info/warn/error/critical)";
        return false;
    }

    // runtime=0（循环模式）时 loop 必须大于 0
    int rt = std::stoi(values_.at("runtime"));
    int lp = std::stoi(values_.at("loop"));
    if (rt == 0 && lp == 0) {
        err = "loop must be > 0 when runtime=0 (loop mode)";
        return false;
    }

    // 值缓冲池参数必须为正：非正值会在后续分配/注册中产生空区域
    // （buffersize 为字节量留 int64，buffercount 为块数用 int，与 getter 值域一致）
    int64_t bufferSize = std::stoll(values_.at("buffersize"));
    int bufferCount = std::stoi(values_.at("buffercount"));
    if (bufferSize <= 0 || bufferCount <= 0) {
        err = "invalid value buffer pool config: buffersize=" + std::to_string(bufferSize) +
              " buffercount=" + std::to_string(bufferCount);
        return false;
    }
    return true;
}

int ConfigUtil::GetTestMode() const
{
    return std::stoi(values_.at("testmode"));
}

int ConfigUtil::GetRuntime() const
{
    return std::stoi(values_.at("runtime"));
}

int ConfigUtil::GetLoop() const
{
    return std::stoi(values_.at("loop"));
}

bool ConfigUtil::GetPerf() const
{
    return values_.at("perf") == "on";
}

bool ConfigUtil::GetOnlyConnect() const
{
    return values_.at("onlyconnect") == "on";
}

int ConfigUtil::GetStartCoreId() const
{
    return std::stoi(values_.at("startcoreid"));
}

int ConfigUtil::GetNumJobs() const
{
    return std::stoi(values_.at("numjobs"));
}

int64_t ConfigUtil::GetBufferSize() const
{
    return std::stoll(values_.at("buffersize"));
}

int ConfigUtil::GetBufferCount() const
{
    return std::stoi(values_.at("buffercount"));
}

std::string ConfigUtil::GetBufferType() const
{
    return values_.at("buffertype");
}

int ConfigUtil::GetDeviceId() const
{
    return std::stoi(values_.at("deviceid"));
}

std::string ConfigUtil::GetReadLogDir() const
{
    return values_.at("readlogdir");
}

std::string ConfigUtil::GetLogDir() const
{
    return values_.at("logdir");
}

spdlog::level::level_enum ConfigUtil::GetLogLevel() const
{
    spdlog::level::level_enum level = spdlog::level::info;
    (void)StringToLevel(values_.at("log.level"), level);
    return level;
}

std::string ConfigUtil::GetLogPattern() const
{
    return values_.at("log.pattern");
}

std::string ConfigUtil::GetLogFileName() const
{
    return values_.at("log.file_name");
}

std::string ConfigUtil::GetLogWarnFileName() const
{
    return values_.at("log.warn_file_name");
}

std::string ConfigUtil::GetLogErrorFileName() const
{
    return values_.at("log.error_file_name");
}

int64_t ConfigUtil::GetLogMaxFileSize() const
{
    return std::stoll(values_.at("log.max_file_size"));
}

int ConfigUtil::GetLogMaxFiles() const
{
    return std::stoi(values_.at("log.max_files"));
}

int ConfigUtil::GetLogAsyncQueueSize() const
{
    return std::stoi(values_.at("log.async_queue_size"));
}

int ConfigUtil::GetLogAsyncFlushIntervalMs() const
{
    return std::stoi(values_.at("log.async_flush_interval_ms"));
}

void ConfigUtil::Print() const
{
    LOG_INFO("testmode={}", GetTestMode());
    LOG_INFO("runtime={}", GetRuntime());
    LOG_INFO("loop={}", GetLoop());
    LOG_INFO("perf={}", GetPerf());
    LOG_INFO("onlyconnect={}", GetOnlyConnect());
    LOG_INFO("startcoreid={}", GetStartCoreId());
    LOG_INFO("numjobs={}", GetNumJobs());
    LOG_INFO("buffersize={}", GetBufferSize());
    LOG_INFO("buffercount={}", GetBufferCount());
    LOG_INFO("buffertype={}", GetBufferType());
    LOG_INFO("deviceid={}", GetDeviceId());
    LOG_INFO("readlogdir={}", GetReadLogDir());
    LOG_INFO("logdir={}", GetLogDir());

    LOG_INFO("log.level={}", spdlog::level::to_string_view(GetLogLevel()));
    LOG_INFO("log.pattern={}", GetLogPattern());
    LOG_INFO("log.file_name={}", GetLogFileName());
    LOG_INFO("log.warn_file_name={}", GetLogWarnFileName());
    LOG_INFO("log.error_file_name={}", GetLogErrorFileName());
    LOG_INFO("log.max_file_size={}", GetLogMaxFileSize());
    LOG_INFO("log.max_files={}", GetLogMaxFiles());
    LOG_INFO("log.async_queue_size={}", GetLogAsyncQueueSize());
    LOG_INFO("log.async_flush_interval_ms={}", GetLogAsyncFlushIntervalMs());
}
