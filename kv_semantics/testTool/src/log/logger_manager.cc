/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "log/logger_manager.h"
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <spdlog/details/thread_pool.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <sstream>
#include <vector>
#include "utils/config_util.h"
#include "utils/file_util.h"

namespace {

/** 请求明细日志文件名模板（strftime 标记按初始化时刻展开，与全量日志命名风格一致） */
constexpr char kReqLogFileNameTpl[] = "req_%Y%m%d_%H.log";

/** 按给定时刻展开文件名模板中的 strftime 时间标记 */
std::string ExpandFileName(const std::string &tpl, std::time_t t)
{
    const std::tm *tmNow = std::localtime(&t);
    if (tmNow == nullptr) { return tpl; }
    std::ostringstream oss;
    oss << std::put_time(tmNow, tpl.c_str());
    return oss.str();
}

}  // namespace

bool LoggerManager::Init(const ConfigUtil &config, std::string &err)
{
    // 重复初始化时先释放旧资源
    if (logger_ != nullptr) { Shutdown(); }

    if (!FileUtil::EnsureDir(config.GetLogDir(), err)) { return false; }
    const std::filesystem::path dirPath(config.GetLogDir());

    // 线程池 / sink / logger 的构造与注册可能因日志目录不可写、磁盘满等环境问题抛异常，
    // 统一捕获转为失败返回，避免异常逃逸导致进程 terminate
    try {
        threadPool_ = std::make_shared<spdlog::details::thread_pool>(
            static_cast<size_t>(config.GetLogAsyncQueueSize()), 1);

        // 三个原生 rotating sink；文件名已由 ConfigUtil 按加载时刻展开时间标记，
        // warn / error 专用 sink 用阈值过滤级别
        auto maxSize = static_cast<size_t>(config.GetLogMaxFileSize());
        auto maxFiles = static_cast<size_t>(config.GetLogMaxFiles());
        std::vector<spdlog::sink_ptr> sinks;
        sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            (dirPath / config.GetLogFileName()).string(), maxSize, maxFiles));

        auto warnSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            (dirPath / config.GetLogWarnFileName()).string(), maxSize, maxFiles);
        warnSink->set_level(spdlog::level::warn);
        sinks.push_back(warnSink);

        auto errorSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            (dirPath / config.GetLogErrorFileName()).string(), maxSize, maxFiles);
        errorSink->set_level(spdlog::level::err);
        sinks.push_back(errorSink);

        // 控制台 sink（log.console，缺省 on）：无色纯文本（输出常被重定向到文件，不掺 ANSI
        // 色码），随主 logger 级别（log.level）过滤，与文件同款 pattern
        if (config.GetLogConsole()) {
            sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_mt>());
        }

        // register_logger 会按 registry 默认值覆盖格式与级别，自定义设置须放在注册之后
        logger_ = std::make_shared<spdlog::async_logger>(
            "main", sinks.begin(), sinks.end(), threadPool_, spdlog::async_overflow_policy::block);
        spdlog::register_logger(logger_);

        // 请求明细日志：put / get / exist / delete 任务的明细行共用一个追加文件；
        // 仅 perf=off 时创建（perf=on 压测模式关闭明细输出，GetReqLogger 返回 nullptr，
        // LOG_INFO_REQ 宏与各明细日志函数的空指针路径安全跳过），readlogdir 目录随创建
        // 一并保证存在；与全量日志完全同款配置（pattern / 级别沿用 log.pattern /
        // log.level，不单独定制），仅 sink 换为 readlogdir 下的追加文件；共用线程池
        if (!config.GetPerf()) {
            if (!FileUtil::EnsureDir(config.GetReadLogDir(), err)) {
                Shutdown();
                return false;
            }
            const std::filesystem::path reqDirPath(config.GetReadLogDir());
            const std::string reqPath =
                (reqDirPath / ExpandFileName(kReqLogFileNameTpl, std::time(nullptr))).string();
            // 第二参数 truncate=false：多轮运行追加同一文件（与原语义一致），不截断
            auto reqSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(reqPath, false);
            reqLogger_ = std::make_shared<spdlog::async_logger>(
                "req", reqSink, threadPool_, spdlog::async_overflow_policy::block);
            spdlog::register_logger(reqLogger_);
        }
    } catch (const std::exception &e) {
        err = e.what();
        Shutdown();
        return false;
    }

    // pattern / 级别为 logger 默认配置（conf 的 log.pattern / log.level），主 logger 与
    // 请求明细 logger 同款应用；主 logger 级别始终取配置的 log.level（perf 不再强制
    // warn：压测时 info 级的指标与运行信息照常输出，压测的日志开销削减由请求明细日志
    // 关闭承担）；请求明细 logger 仅 perf=off 存在，直接用配置值
    try {
        logger_->set_pattern(config.GetLogPattern());
        if (reqLogger_ != nullptr) { reqLogger_->set_pattern(config.GetLogPattern()); }
    } catch (const std::exception &e) {
        err = "invalid log.pattern: ";
        err += e.what();
        Shutdown();
        return false;
    }

    logger_->set_level(config.GetLogLevel());
    if (reqLogger_ != nullptr) { reqLogger_->set_level(config.GetLogLevel()); }
    logger_->flush_on(spdlog::level::err);  // error 及以上立即落盘
    spdlog::flush_every(std::chrono::milliseconds(config.GetLogAsyncFlushIntervalMs()));
    return true;
}

void LoggerManager::Shutdown()
{
    if (logger_ != nullptr) { logger_->flush(); }
    if (reqLogger_ != nullptr) { reqLogger_->flush(); }

    spdlog::shutdown();
    // 必须先停线程池再销毁 logger：worker 处理队列尾部消息时还要通过 weak_ptr 取到 logger
    threadPool_.reset();
    logger_.reset();
    reqLogger_.reset();
}

std::shared_ptr<spdlog::logger> LoggerManager::GetLogger() { return logger_; }

std::shared_ptr<spdlog::logger> LoggerManager::GetReqLogger() { return reqLogger_; }

// 请求明细日志的写入经 LOG_INFO_REQ 宏直接走 GetReqLogger（与 LOG_xxx 走 GetLogger 完全
// 对称），不再需要单独的写入接口；async_logger 入队线程安全，单写线程保证明细文件内行完整有序
