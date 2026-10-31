/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Agent日志模块实现
*/

// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#include "AgentLogger.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <system_error>
#include <vector>

// Weak stub: when the agent is loaded into a process that already provides
// GetAgentLogFunc (e.g. kvcache_pool linked into the host), the strong
// definition there overrides this one. When the agent runs standalone
// (smoke tests), this returns nullptr and AGENT_LOG falls back to the
// file/console logger.
extern "C" __attribute__((weak)) AgentlogFunc GetAgentLogFunc(void)
{
	return nullptr;
}

namespace agent {

// 动态获取 GetDplogFunc 函数指针
AgentlogFunc get_agentlog_func_impl(void)
{
	return GetAgentLogFunc();   // 返回 dplog
}

namespace {

bool is_permission_error(const std::error_code& error)
{
	return error == std::errc::permission_denied ||
		error == std::errc::operation_not_permitted;
}

const char *module_basename(const char *module)
{
	const char *base = module;
	for (const char *cursor = module; *cursor != '\0'; ++cursor) {
		if (*cursor == '/' || *cursor == '\\') {
			base = cursor + 1;
		}
	}
	return base;
}

std::string current_time()
{
	const auto now = std::chrono::system_clock::now();
	const std::time_t value = std::chrono::system_clock::to_time_t(now);
	std::tm local_time{};
	localtime_r(&value, &local_time);
	std::ostringstream out;
	out << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S");
	return out.str();
}

std::string format_line(LogLevel level, const char *module,
				const char *interface, const std::string& message)
{
	return "[" + std::string(log_level_name(level)) + "][" + current_time() +
		"][" + module_basename(module) + "][" + interface + "] " + message;
}

} // namespace

const char *log_level_name(LogLevel level)
{
    switch (level) {
        case LogLevel::INFO: return "INFO";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
    }
    return "ERROR";
}

// 日志初始化需要处理多种目录/权限异常分支, 复杂度不可避免
// NOLINTBEGIN(huge_method, huge_cyclomatic_complexity)
int AgentLogger::init(const char *base_dir, const std::string& component)
{
    if (output.is_open()) {
        output.close();
    }
    output.clear();
    if (!base_dir || base_dir[0] == '\0') {
        return 0;
    }

    std::error_code error;
    const auto base_directory = std::filesystem::path(base_dir);
    const auto base_status = std::filesystem::status(base_directory, error);
    if (error) {
        AGENT_LOG(this, ::agent::LogLevel::ERROR,
                                        "failed to inspect log directory " +
                                            base_directory.string() +
                                            ": " + error.message());
        if (error == std::errc::no_such_file_or_directory) {
            return 0;
        }
        if (is_permission_error(error)) {
            return 0;
        }
        return AGENT_LOG_ERR_DIRECTORY_READ_FAILED;
    }
    if (!std::filesystem::exists(base_status)) {
        AGENT_LOG(this, ::agent::LogLevel::ERROR,
                            "log directory does not exist: " + base_directory.string());
        return 0;
    }
    if (!std::filesystem::is_directory(base_status)) {
        AGENT_LOG(this, ::agent::LogLevel::ERROR,
                            "log path is not a directory: " + base_directory.string());
        return AGENT_LOG_ERR_PATH_NOT_DIRECTORY;
    }

    const auto directory = base_directory / component;
    const auto status = std::filesystem::status(directory, error);
    if (error && error != std::errc::no_such_file_or_directory) {
        AGENT_LOG(this, ::agent::LogLevel::ERROR,
                            "failed to inspect component log directory " +
                                directory.string() + ": " + error.message());
        if (is_permission_error(error)) {
            return AGENT_LOG_ERR_PERMISSION_DENIED;
        }
        return AGENT_LOG_ERR_DIRECTORY_READ_FAILED;
    }
    if (error == std::errc::no_such_file_or_directory ||
            !std::filesystem::exists(status)) {
        error.clear();
        std::filesystem::create_directory(directory, error);
        if (error) {
            AGENT_LOG(this, ::agent::LogLevel::ERROR,
                                "failed to create component log directory " +
                                    directory.string() + ": " + error.message());
            if (is_permission_error(error)) {
                return AGENT_LOG_ERR_PERMISSION_DENIED;
            }
            return AGENT_LOG_ERR_COMPONENT_CREATE_FAILED;
        }
    } else if (!std::filesystem::is_directory(status)) {
        AGENT_LOG(this, ::agent::LogLevel::ERROR,
                            "component log path is not a directory: " +
                                directory.string());
        return AGENT_LOG_ERR_PATH_NOT_DIRECTORY;
    }
    return open_next_log(directory, component);
}
// NOLINTEND(huge_method, huge_cyclomatic_complexity)

// 日志轮转逻辑需遍历+排序+清理+创建, 分支较多
// NOLINTBEGIN(huge_method, huge_cyclomatic_complexity)
int AgentLogger::open_next_log(const std::filesystem::path& directory,
                              const std::string& component)
{
    const std::regex pattern("^" + component + R"(\.log\.([0-9]+)$)");
    std::vector<std::pair<uint64_t, std::filesystem::path>> logs;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
        std::smatch match;
        const auto name = entry.path().filename().string();
        if (entry.is_regular_file() &&
                std::regex_match(name.cbegin(), name.cend(), match, pattern)) {
            try {
                logs.emplace_back(std::stoull(match[1].str()), entry.path());
            } catch (const std::exception&) {
                // Ignore names whose numeric suffix cannot be represented.
            }
        }
    }
    if (error) {
        AGENT_LOG(this, ::agent::LogLevel::ERROR,
                                        "failed to inspect log directory " + directory.string() +
                                            ": " + error.message());
        if (error == std::errc::no_such_file_or_directory) {
            return AGENT_LOG_ERR_DIRECTORY_NOT_FOUND;
        }
        if (is_permission_error(error)) {
            return AGENT_LOG_ERR_PERMISSION_DENIED;
        }
        return AGENT_LOG_ERR_DIRECTORY_READ_FAILED;
    }

    std::sort(logs.begin(), logs.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });
    const uint64_t next = logs.empty() ? 1 : logs.back().first + 1;
    while (logs.size() >= 20) {
        std::filesystem::remove(logs.front().second, error);
        if (error) {
            AGENT_LOG(this, ::agent::LogLevel::ERROR,
                                            "failed to remove old log " +
                                                logs.front().second.string() + ": " + error.message());
            if (is_permission_error(error)) {
                return AGENT_LOG_ERR_PERMISSION_DENIED;
            }
            return AGENT_LOG_ERR_ROTATION_FAILED;
        }
        logs.erase(logs.begin());
    }

    const auto path = directory / (component + ".log." + std::to_string(next));
    output.open(path, std::ios::out | std::ios::trunc);
    if (!output) {
        const int error_number = errno ? errno : EIO;
        AGENT_LOG(this, ::agent::LogLevel::ERROR,
                            "failed to open log file " + path.string());
        if (error_number == EACCES || error_number == EPERM) {
            return AGENT_LOG_ERR_PERMISSION_DENIED;
        }
        return AGENT_LOG_ERR_FILE_CREATE_FAILED;
    }
    return 0;
}
// NOLINTEND(huge_method, huge_cyclomatic_complexity)

void AgentLogger::log(LogLevel level, const char *module,
                                         const char *interface, const std::string& message)
{
    std::lock_guard guard(lock);
    std::ostream& console = level == LogLevel::ERROR ? std::cerr : std::cout;
    const std::string line = format_line(level, module, interface, message);
    console << line << std::endl;
    // to file (if base_dir = nullptr/""/dir not exist, will not write to file)
    if (output) {
        output << line << std::endl;
    }
}

void log(AgentLogger *logger, LogLevel level, const char *module,
                 const char *interface, const std::string& message)
{
    // to file & console
    if (logger) {
        logger->log(level, module, interface, message);
        return;
    }

    // to console only (AGENT macro explicitly specified to disable log writing)
    std::ostream& console = level == LogLevel::ERROR ? std::cerr : std::cout;
    console << format_line(level, module, interface, message) << std::endl;
}

} // namespace agent
