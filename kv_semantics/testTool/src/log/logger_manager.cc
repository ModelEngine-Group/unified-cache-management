/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "log/logger_manager.h"

#include <chrono>
#include <filesystem>
#include <vector>

#include <spdlog/details/thread_pool.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include "utils/config_util.h"
#include "utils/file_util.h"

bool LoggerManager::Init(std::string &err)
{
    // 重复初始化时先释放旧资源
    if (logger_ != nullptr) {
        Shutdown();
    }

    if (!FileUtil::EnsureDir(ConfigUtil::GetLogDir(), err)) {
        return false;
    }
    const std::filesystem::path dirPath(ConfigUtil::GetLogDir());

    threadPool_ = std::make_shared<spdlog::details::thread_pool>(
        static_cast<size_t>(ConfigUtil::GetLogAsyncQueueSize()), 1);

    // 三个原生 rotating sink；文件名已由 ConfigUtil 按加载时刻展开时间标记，
    // warn / error 专用 sink 用阈值过滤级别
    auto maxSize = static_cast<size_t>(ConfigUtil::GetLogMaxFileSize());
    auto maxFiles = static_cast<size_t>(ConfigUtil::GetLogMaxFiles());
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        (dirPath / ConfigUtil::GetLogFileName()).string(), maxSize, maxFiles));

    auto warnSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        (dirPath / ConfigUtil::GetLogWarnFileName()).string(), maxSize, maxFiles);
    warnSink->set_level(spdlog::level::warn);
    sinks.push_back(warnSink);

    auto errorSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        (dirPath / ConfigUtil::GetLogErrorFileName()).string(), maxSize, maxFiles);
    errorSink->set_level(spdlog::level::err);
    sinks.push_back(errorSink);

    // register_logger 会按 registry 默认值覆盖格式与级别，自定义设置须放在注册之后
    logger_ = std::make_shared<spdlog::async_logger>("main", sinks.begin(), sinks.end(),
                                                     threadPool_,
                                                     spdlog::async_overflow_policy::block);
    spdlog::register_logger(logger_);

    try {
        logger_->set_pattern(ConfigUtil::GetLogPattern());
    } catch (const std::exception &e) {
        err = "log.pattern 非法: ";
        err += e.what();
        Shutdown();
        return false;
    }

    logger_->set_level(ConfigUtil::GetLogLevel());
    logger_->flush_on(spdlog::level::err); // error 及以上立即落盘
    spdlog::flush_every(std::chrono::milliseconds(ConfigUtil::GetLogAsyncFlushIntervalMs()));
    return true;
}

void LoggerManager::Shutdown()
{
    if (logger_ != nullptr) {
        logger_->flush();
    }

    spdlog::shutdown();
    // 必须先停线程池再销毁 logger：worker 处理队列尾部消息时还要通过 weak_ptr 取到 logger
    threadPool_.reset();
    logger_.reset();
}

std::shared_ptr<spdlog::logger> LoggerManager::GetLogger()
{
    return logger_;
}
