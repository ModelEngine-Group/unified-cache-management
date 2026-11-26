/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */

#include "log/logger_manager.h"
#include <chrono>
#include <ctime>
#include <filesystem>
#include <spdlog/details/thread_pool.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <stdexcept>
#include <vector>
#include "utils/file_util.h"

bool LoggerManager::Init(const LogConfigUtil &logConfig, std::string &err)
{
    // 重复初始化直接报错拒绝
    if (logger_ != nullptr) {
        err = "logger already initialized";
        return false;
    }

    // 级别名转枚举（from_str 对未知名返回 off，显式 off 合法）
    const std::string levelName = logConfig.GetLevel();
    const spdlog::level::level_enum level = spdlog::level::from_str(levelName);
    if (level == spdlog::level::off && levelName != "off") {
        err =
            "invalid log.level: " + levelName + " (expected trace/debug/info/warn/error/critical)";
        return false;
    }

    // 三个文件名模板按启动时刻展开时间标记
    const std::time_t now = std::time(nullptr);
    const std::tm tmNow = *std::localtime(&now);
    const std::string mainName = spdlog::sinks::daily_filename_format_calculator::calc_filename(
        logConfig.GetFileName(), tmNow);
    const std::string getName = spdlog::sinks::daily_filename_format_calculator::calc_filename(
        logConfig.GetGetFileName(), tmNow);
    const std::string otherName = spdlog::sinks::daily_filename_format_calculator::calc_filename(
        logConfig.GetOtherFileName(), tmNow);

    // 各创建环节失败统一抛异常，此处一次收口：填 err 并自清理
    try {
        threadPool_ =
            std::make_shared<spdlog::details::thread_pool>(logConfig.GetAsyncQueueSize(), 1);
        CreateLogger(logConfig, level, "main", logConfig.GetDir(), mainName, logger_);
        CreateLogger(logConfig, level, "get", logConfig.GetGetDir(), getName, getLogger_);
        CreateLogger(logConfig, level, "other", logConfig.GetOtherDir(), otherName, otherLogger_);
        spdlog::flush_every(std::chrono::milliseconds(logConfig.GetAsyncFlushIntervalMs()));
    } catch (const std::exception &e) {
        err = e.what();
        Shutdown();
        return false;
    }
    return true;
}

void LoggerManager::CreateLogger(const LogConfigUtil &logConfig, spdlog::level::level_enum level,
                                 const char *name, const std::string &dir,
                                 const std::string &fileName, std::shared_ptr<spdlog::logger> &out)
{
    std::string dirErr;
    if (!FileUtil::EnsureDir(dir, dirErr)) {
        throw std::runtime_error("log dir '" + dir + "': " + dirErr);
    }
    std::vector<spdlog::sink_ptr> sinks;
    // rotating 文件 sink：单文件超限滚动清理
    sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        (std::filesystem::path(dir) / fileName).string(), logConfig.GetMaxFileSize(),
        logConfig.GetMaxFiles()));
    // 控制台 sink（无色纯文本，输出常被重定向）
    if (logConfig.GetConsole()) {
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_mt>());
    }
    out = std::make_shared<spdlog::async_logger>(name, sinks.begin(), sinks.end(), threadPool_,
                                                 spdlog::async_overflow_policy::block);
    spdlog::register_logger(out);
    out->set_pattern(logConfig.GetPattern());
    out->set_level(level);
}

void LoggerManager::Shutdown()
{
    if (logger_ != nullptr) { logger_->flush(); }
    if (getLogger_ != nullptr) { getLogger_->flush(); }
    if (otherLogger_ != nullptr) { otherLogger_->flush(); }

    spdlog::shutdown();
    // 必须先停线程池再销毁 logger：worker 处理队列尾部消息时还要通过 weak_ptr 取到 logger
    threadPool_.reset();
    logger_.reset();
    getLogger_.reset();
    otherLogger_.reset();
}

void LoggerManager::ShutdownForExit()
{
    // 进程退出专用（见头文件注释）：只回收自有资源，spdlog::shutdown 会 join 库侧
    // 不退出的线程导致挂死
    if (logger_ != nullptr) { logger_->flush(); }
    if (getLogger_ != nullptr) { getLogger_->flush(); }
    if (otherLogger_ != nullptr) { otherLogger_->flush(); }
    threadPool_.reset();
    logger_.reset();
    getLogger_.reset();
    otherLogger_.reset();
}
