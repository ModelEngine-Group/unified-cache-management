/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <memory>
#include <string>

#include <spdlog/async_logger.h>
#include <spdlog/spdlog.h>

/** 日志管理器：异步日志的初始化 / 获取 / 关闭入口；滚动与级别过滤均用 spdlog 原生能力 */
class LoggerManager {
public:
    /** 从 ConfigUtil 读取 log 参数并初始化异步日志；失败填充 err */
    static bool Init(std::string &err);
    /** 冲刷并释放日志资源；队列残留消息全部落盘后线程池才退出，可重复调用 */
    static void Shutdown();
    /** 获取全局 logger；未初始化或已关闭时返回 nullptr */
    static std::shared_ptr<spdlog::logger> GetLogger();

    LoggerManager() = delete; // 静态工具类，禁止实例化

private:
    /** 异步日志线程池（单写线程保证写入顺序） */
    inline static std::shared_ptr<spdlog::details::thread_pool> threadPool_;
    /** 全局异步 logger（注册名 "main"） */
    inline static std::shared_ptr<spdlog::logger> logger_;
};

// ============================================================================
// 日志输出宏：统一出口；未初始化或已关闭时安全跳过。格式语法同 fmt。
// ============================================================================
#define LOG_IMPL(LEVEL, ...)                                                                       \
    do {                                                                                           \
        std::shared_ptr<spdlog::logger> logLg_ = ::LoggerManager::GetLogger();                     \
        if (logLg_ != nullptr) {                                                                   \
            logLg_->log(LEVEL, __VA_ARGS__);                                                       \
        }                                                                                          \
    } while (0)

/** 输出 debug 级别日志（fmt 格式） */
#define LOG_DEBUG(...) LOG_IMPL(spdlog::level::debug, __VA_ARGS__)
/** 输出 info 级别日志（fmt 格式） */
#define LOG_INFO(...) LOG_IMPL(spdlog::level::info, __VA_ARGS__)
/** 输出 warn 级别日志（fmt 格式） */
#define LOG_WARN(...) LOG_IMPL(spdlog::level::warn, __VA_ARGS__)
/** 输出 error 级别日志（fmt 格式） */
#define LOG_ERROR(...) LOG_IMPL(spdlog::level::err, __VA_ARGS__)
