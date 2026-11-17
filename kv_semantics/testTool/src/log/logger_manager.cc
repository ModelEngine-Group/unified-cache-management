/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "log/logger_manager.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>

#include <spdlog/details/thread_pool.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include "utils/config_util.h"
#include "utils/file_util.h"

namespace {

/** get 读回日志文件名模板（strftime 标记按初始化时刻展开，与全量日志命名风格一致） */
constexpr char kGetLogFileNameTpl[] = "get_%Y%m%d_%H.log";

/** 按给定时刻展开文件名模板中的 strftime 时间标记 */
std::string ExpandFileName(const std::string &tpl, std::time_t t)
{
    const std::tm *tmNow = std::localtime(&t);
    if (tmNow == nullptr) {
        return tpl;
    }
    std::ostringstream oss;
    oss << std::put_time(tmNow, tpl.c_str());
    return oss.str();
}

}  // namespace

bool LoggerManager::Init(const ConfigUtil &config, std::string &err)
{
    // 重复初始化时先释放旧资源
    if (logger_ != nullptr) {
        Shutdown();
    }

    if (!FileUtil::EnsureDir(config.GetLogDir(), err)) {
        return false;
    }
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

        // register_logger 会按 registry 默认值覆盖格式与级别，自定义设置须放在注册之后
        logger_ = std::make_shared<spdlog::async_logger>("main", sinks.begin(), sinks.end(),
                                                        threadPool_,
                                                        spdlog::async_overflow_policy::block);
        spdlog::register_logger(logger_);

        // get 读回日志（perf=off 启用）：readlogdir 下追加文件、纯内容行（%v 不带
        // pattern 前缀），与全量日志共用线程池；perf=on 不创建，写入口安全跳过
        if (!config.GetPerf()) {
            if (!FileUtil::EnsureDir(config.GetReadLogDir(), err)) {
                Shutdown();
                return false;
            }
            const std::filesystem::path getDirPath(config.GetReadLogDir());
            const std::string getPath =
                (getDirPath / ExpandFileName(kGetLogFileNameTpl, std::time(nullptr))).string();
            // 第二参数 truncate=false：多轮运行追加同一文件（与原语义一致），不截断
            auto getSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(getPath, false);
            getLogger_ = std::make_shared<spdlog::async_logger>(
                "get", getSink, threadPool_, spdlog::async_overflow_policy::block);
            spdlog::register_logger(getLogger_);
            getLogger_->set_pattern("%v");
        }
    } catch (const std::exception &e) {
        err = e.what();
        Shutdown();
        return false;
    }

    try {
        logger_->set_pattern(config.GetLogPattern());
    } catch (const std::exception &e) {
        err = "invalid log.pattern: ";
        err += e.what();
        Shutdown();
        return false;
    }

    logger_->set_level(config.GetLogLevel());
    logger_->flush_on(spdlog::level::err); // error 及以上立即落盘
    spdlog::flush_every(std::chrono::milliseconds(config.GetLogAsyncFlushIntervalMs()));
    return true;
}

void LoggerManager::Shutdown()
{
    if (logger_ != nullptr) {
        logger_->flush();
    }
    if (getLogger_ != nullptr) {
        getLogger_->flush();
    }

    spdlog::shutdown();
    // 必须先停线程池再销毁 logger：worker 处理队列尾部消息时还要通过 weak_ptr 取到 logger
    threadPool_.reset();
    logger_.reset();
    getLogger_.reset();
}

std::shared_ptr<spdlog::logger> LoggerManager::GetLogger()
{
    return logger_;
}

bool LoggerManager::IsGetLogEnabled()
{
    return getLogger_ != nullptr;
}

void LoggerManager::WriteGetLog(const std::string &line)
{
    // 未启用（perf=on）或已关闭时安全跳过；async_logger 入队线程安全，
    // 单写线程保证 get 文件内行完整有序
    if (getLogger_ == nullptr) {
        return;
    }
    getLogger_->info("{}", line);
}
