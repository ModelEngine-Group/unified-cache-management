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
            err = "配置第 " + std::to_string(lineNo) + " 行缺少 '=': " + trimmed;
            return false;
        }
        std::string key = Trim(trimmed.substr(0, sep));
        std::string value = Trim(trimmed.substr(sep + 1));
        size_t hash = value.find('#');
        if (hash != std::string::npos) {
            value = Trim(value.substr(0, hash));
        }
        if (key.empty()) {
            err = "配置第 " + std::to_string(lineNo) + " 行 key 为空";
            return false;
        }
        if (!items.emplace(key, value).second) {
            err = "配置第 " + std::to_string(lineNo) + " 行 key 重复: " + key;
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
        err = "log.file_name / log.warn_file_name / log.error_file_name 展开后文件名重复";
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
        err = "log.level 非法: " + values_.at("log.level") +
              "（可选 trace/debug/info/warn/error/critical）";
        return false;
    }

    // runtime=0（循环模式）时 loop 必须大于 0
    int64_t rt = std::stoll(values_.at("runtime"));
    int64_t lp = std::stoll(values_.at("loop"));
    if (rt == 0 && lp == 0) {
        err = "runtime=0（循环模式）时 loop 必须大于 0";
        return false;
    }
    return true;
}

int64_t ConfigUtil::GetTestMode()
{
    return std::stoll(values_.at("testmode"));
}

int64_t ConfigUtil::GetRuntime()
{
    return std::stoll(values_.at("runtime"));
}

int64_t ConfigUtil::GetLoop()
{
    return std::stoll(values_.at("loop"));
}

bool ConfigUtil::GetPerf()
{
    return values_.at("perf") == "on";
}

bool ConfigUtil::GetOnlyConnect()
{
    return values_.at("onlyconnect") == "on";
}

int64_t ConfigUtil::GetStartCoreId()
{
    return std::stoll(values_.at("startcoreid"));
}

int64_t ConfigUtil::GetNumJobs()
{
    return std::stoll(values_.at("numjobs"));
}

std::string ConfigUtil::GetReadLogDir()
{
    return values_.at("readlogdir");
}

std::string ConfigUtil::GetLogDir()
{
    return values_.at("logdir");
}

spdlog::level::level_enum ConfigUtil::GetLogLevel()
{
    spdlog::level::level_enum level = spdlog::level::info;
    (void)StringToLevel(values_.at("log.level"), level);
    return level;
}

std::string ConfigUtil::GetLogPattern()
{
    return values_.at("log.pattern");
}

std::string ConfigUtil::GetLogFileName()
{
    return values_.at("log.file_name");
}

std::string ConfigUtil::GetLogWarnFileName()
{
    return values_.at("log.warn_file_name");
}

std::string ConfigUtil::GetLogErrorFileName()
{
    return values_.at("log.error_file_name");
}

int64_t ConfigUtil::GetLogMaxFileSize()
{
    return std::stoll(values_.at("log.max_file_size"));
}

int64_t ConfigUtil::GetLogMaxFiles()
{
    return std::stoll(values_.at("log.max_files"));
}

int64_t ConfigUtil::GetLogAsyncQueueSize()
{
    return std::stoll(values_.at("log.async_queue_size"));
}

int64_t ConfigUtil::GetLogAsyncFlushIntervalMs()
{
    return std::stoll(values_.at("log.async_flush_interval_ms"));
}

void ConfigUtil::Print()
{
    LOG_INFO("testmode={}", GetTestMode());
    LOG_INFO("runtime={}", GetRuntime());
    LOG_INFO("loop={}", GetLoop());
    LOG_INFO("perf={}", GetPerf());
    LOG_INFO("onlyconnect={}", GetOnlyConnect());
    LOG_INFO("startcoreid={}", GetStartCoreId());
    LOG_INFO("numjobs={}", GetNumJobs());
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
