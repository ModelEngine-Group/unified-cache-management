/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <memory>
#include <spdlog/async_logger.h>
#include <spdlog/spdlog.h>
#include <string>

class ConfigUtil;

/**
 * 日志管理器：异步日志的初始化 / 获取 / 关闭入口；统一管理两类 logger——
 * 全量日志（滚动文件 + 可选控制台 sink，LOG_xxx 宏输出）与请求明细日志（perf=off 时
 * put/get/exist/delete 任务的明细行，LOG_INFO_REQ 宏输出），两者配置完全同款
 * （pattern / 级别沿用 log.pattern / log.level），仅 sink 不同（请求明细落 readlogdir
 * 追加文件），共用同一异步线程池；两类 logger 获取方式对称（GetLogger / GetReqLogger）。
 */
class LoggerManager {
public:
    /**
     * 从传入的配置实例读取 log 参数并初始化异步日志（显式注入，不访问配置单例）；
     * 全量日志的 sink 随配置组装（log.console=on 时附控制台 sink）；请求明细日志仅
     * perf=off 时创建（perf=on 压测模式关闭明细输出）；失败填充 err（内部已自清理）
     */
    static bool Init(const ConfigUtil &config, std::string &err);
    /** 冲刷并释放全部日志资源；队列残留消息全部落盘后线程池才退出，可重复调用 */
    static void Shutdown();
    /** 获取全局 logger；未初始化或已关闭时返回 nullptr */
    static std::shared_ptr<spdlog::logger> GetLogger();
    /** 获取请求明细日志 logger；未初始化（perf=on）或已关闭时返回 nullptr */
    static std::shared_ptr<spdlog::logger> GetReqLogger();

    LoggerManager() = delete;  // 静态工具类，禁止实例化

private:
    /** 异步日志线程池（单写线程保证各文件内写入顺序） */
    inline static std::shared_ptr<spdlog::details::thread_pool> threadPool_;
    /** 全局异步 logger（注册名 "main"） */
    inline static std::shared_ptr<spdlog::logger> logger_;
    /** 请求明细日志异步 logger（注册名 "req"；perf=on 时不创建，保持 nullptr） */
    inline static std::shared_ptr<spdlog::logger> reqLogger_;
};

// ============================================================================
// 日志输出宏：统一出口；未初始化或已关闭时安全跳过。格式语法同 fmt。
// ============================================================================
#define LOG_IMPL(LEVEL, ...)                                                   \
    do {                                                                       \
        std::shared_ptr<spdlog::logger> logLg_ = ::LoggerManager::GetLogger(); \
        if (logLg_ != nullptr) { logLg_->log(LEVEL, __VA_ARGS__); }            \
    } while (0)

/** 输出 debug 级别日志（fmt 格式） */
#define LOG_DEBUG(...) LOG_IMPL(spdlog::level::debug, __VA_ARGS__)
/** 输出 info 级别日志（fmt 格式） */
#define LOG_INFO(...) LOG_IMPL(spdlog::level::info, __VA_ARGS__)
/** 输出 warn 级别日志（fmt 格式） */
#define LOG_WARN(...) LOG_IMPL(spdlog::level::warn, __VA_ARGS__)
/** 输出 error 级别日志（fmt 格式） */
#define LOG_ERROR(...) LOG_IMPL(spdlog::level::err, __VA_ARGS__)

/** 输出一行请求明细日志（fmt 格式，落 readlogdir 独立目录；未启用或已关闭时安全跳过） */
#define LOG_INFO_REQ(...)                                                         \
    do {                                                                          \
        std::shared_ptr<spdlog::logger> reqLg_ = ::LoggerManager::GetReqLogger(); \
        if (reqLg_ != nullptr) { reqLg_->info(__VA_ARGS__); }                     \
    } while (0)
