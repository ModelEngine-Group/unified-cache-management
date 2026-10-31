/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Agent日志模块头文件
*/

// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#pragma once

// File logger initialization errors returned by kv_agent_init().
#define AGENT_LOG_ERR_DIRECTORY_NOT_FOUND (-10000)
#define AGENT_LOG_ERR_PATH_NOT_DIRECTORY (-10001)
#define AGENT_LOG_ERR_PERMISSION_DENIED (-10002)
#define AGENT_LOG_ERR_DIRECTORY_READ_FAILED (-10003)
#define AGENT_LOG_ERR_ROTATION_FAILED (-10004)
#define AGENT_LOG_ERR_FILE_CREATE_FAILED (-10005)
#define AGENT_LOG_ERR_COMPONENT_CREATE_FAILED (-10006)

#ifdef __cplusplus

#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>


// ========== 限频相关定义 ==========

#define NSEC_PER_SEC 1000000000L

#ifndef HZ
#define HZ 100
#endif

// 获取当前时钟周期（基于单调时钟）
static inline unsigned long agent_get_jiffies(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long)(ts.tv_sec * HZ + ts.tv_nsec / (NSEC_PER_SEC / HZ));
}

// 限频状态结构体
struct AgentLogRateLimit {
    unsigned long last_print_jiffies;   // 上次打印时间
    unsigned int print_count;           // 本周期内已打印次数

    AgentLogRateLimit() : last_print_jiffies(0), print_count(0) {}
};

// 限频检查宏
// interval: 时间间隔（秒）
// burst: 初始允许打印次数
// 注意: 由于使用 static 变量, 每个限频点有独立的计数状态
// NOLINTBEGIN(g.PRE.06-cpp)
#define AGENT_CHECK_RATE_LIMIT(interval_sec, burst_count, result_var) \
    do { \
        static AgentLogRateLimit s_rate_limit; \
        unsigned long now = agent_get_jiffies(); \
        unsigned long interval_jiffies = (interval_sec) * HZ; \
        if (now - s_rate_limit.last_print_jiffies >= interval_jiffies) { \
            if (s_rate_limit.print_count < (burst_count)) { \
                result_var = true; \
                s_rate_limit.print_count++; \
            } else { \
                result_var = false; \
            } \
            if (s_rate_limit.print_count >= (burst_count)) { \
                s_rate_limit.last_print_jiffies = now; \
                s_rate_limit.print_count = 0; \
            } \
        } else { \
            result_var = false; \
        } \
    } while (0)
// NOLINTEND(g.PRE.06-cpp)

// ========== 自己声明 kvcache_pool 的类型, 不引用头文件 ==========

// 日志级别
#define LOG_EMERG   0
#define LOG_ALERT   1
#define LOG_CRIT    2
#define LOG_ERR     3
#define LOG_WARNING 4
#define LOG_NOTICE  5
#define LOG_INFO    6
#define LOG_DEBUG   7

// AgentlogFunc 函数指针类型 - 自己定义, 不依赖 kvcache_pool_log.h
// char* 参数匹配 kvcache_pool C ABI, 不可改为 std::string
typedef int (*AgentlogFuncType)(const int mod_id, const int log_level, char *file_name, // NOLINT(g.STD.02-cpp)
                             const int f_line, const char *func_name,
                             const int64_t log_id, const char *format, ...);

using AgentlogFunc = AgentlogFuncType;

// 获取 dplog 函数的声明（由 kvcache_pool 提供, 通过 dlsym 动态获取）
extern "C" AgentlogFunc GetAgentLogFunc(void);

// 日志相关常量
#define PID_KVCACHE_POOL 931
#define LOG_ID_KVCACHE_POOL 0

#define _AGENT_FILENAME_ (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) : __FILE__)

namespace agent {

AgentlogFunc get_agentlog_func_impl(void);

enum class LogLevel {
    INFO,
    DEBUG,
    WARNING,
    ERROR,
};

class AgentLogger {
public:
    AgentLogger() = default;
    ~AgentLogger() = default;
    AgentLogger(const AgentLogger&) = delete;
    AgentLogger& operator=(const AgentLogger&) = delete;
    AgentLogger(AgentLogger&&) = delete;
    AgentLogger& operator=(AgentLogger&&) = delete;

    int init(const char *base_dir, const std::string& component);
    void log(LogLevel level, const char *module, const char *interface,
             const std::string& message);

private:
    int open_next_log(const std::filesystem::path& directory,
                      const std::string& component);
    std::mutex lock;
    std::ofstream output;
};

const char *log_level_name(LogLevel level);
void log(AgentLogger *logger, LogLevel level, const char *module,
         const char *interface, const std::string& message);

} // namespace agent

// NOLINTBEGIN(g.PRE.06-cpp)
#define AGENT_LOG(logger, level, message) \
    do { \
        ::AgentlogFunc _agentLog = ::agent::get_agentlog_func_impl(); \
        if (_agentLog) { \
            int _log_level = (level) == ::agent::LogLevel::ERROR ? LOG_ERR : \
                             (level) == ::agent::LogLevel::WARNING ? LOG_WARNING : \
                             (level) == ::agent::LogLevel::DEBUG ? LOG_DEBUG : LOG_INFO; \
            std::string _log_msg = (message); \
            _agentLog(PID_KVCACHE_POOL, _log_level, (char *)_AGENT_FILENAME_, __LINE__, __func__, \
                   LOG_ID_KVCACHE_POOL, _log_msg.c_str()); \
        } else { \
            ::agent::log((logger), (level), _AGENT_FILENAME_, __func__, (message)); \
        } \
    } while (0)
// NOLINTEND(g.PRE.06-cpp)

// 带限频的 AGENT_LOG 宏
// interval_sec: 时间间隔（秒）, 每隔 interval_sec 秒可打印一次
// burst_count: 初始允许打印次数（一个周期内最多打印 burst_count 次）
// 注意: 每个不同的 interval_sec 和 burst_count 组合有独立的限频状态
#define AGENT_LOG_LIMITED(logger, level, interval_sec, burst_count, message) \
    do { \
        bool _should_print = false; \
        AGENT_CHECK_RATE_LIMIT(interval_sec, burst_count, _should_print); \
        if (_should_print) { \
            AGENT_LOG(logger, level, message); \
        } \
    } while (0)

#endif
