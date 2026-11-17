/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <memory>
#include <string>

#include <spdlog/async_logger.h>
#include <spdlog/spdlog.h>

class ConfigUtil;

/**
 * 日志管理器：异步日志的初始化 / 获取 / 写入 / 关闭入口；统一管理两类 logger——
 * 全量日志（滚动文件、带 pattern 前缀）与 get 读回日志（非性能模式下的读回校验
 * 明细，追加文件、纯内容行），共用同一异步线程池。
 */
class LoggerManager {
public:
    /**
     * 从传入的配置实例读取 log 参数并初始化两类异步日志（显式注入，不访问配置单例）；
     * get 读回日志仅 perf=off 时创建；失败填充 err（内部已自清理）
     */
    static bool Init(const ConfigUtil &config, std::string &err);
    /** 冲刷并释放全部日志资源；队列残留消息全部落盘后线程池才退出，可重复调用 */
    static void Shutdown();
    /** 获取全局 logger；未初始化或已关闭时返回 nullptr */
    static std::shared_ptr<spdlog::logger> GetLogger();
    /** get 读回日志是否启用（perf=off 且初始化成功；已关闭后为 false） */
    static bool IsGetLogEnabled();
    /** 写一行 get 读回日志（内容原样落盘不带 pattern；未启用或已关闭时安全跳过） */
    static void WriteGetLog(const std::string &line);

    LoggerManager() = delete; // 静态工具类，禁止实例化

private:
    /** 异步日志线程池（单写线程保证各文件内写入顺序） */
    inline static std::shared_ptr<spdlog::details::thread_pool> threadPool_;
    /** 全局异步 logger（注册名 "main"） */
    inline static std::shared_ptr<spdlog::logger> logger_;
    /** get 读回日志异步 logger（注册名 "get"；perf=on 时为空） */
    inline static std::shared_ptr<spdlog::logger> getLogger_;
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
