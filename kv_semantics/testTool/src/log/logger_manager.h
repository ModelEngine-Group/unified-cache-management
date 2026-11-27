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

#pragma once

#include <memory>
#include <spdlog/async_logger.h>
#include <spdlog/spdlog.h>
#include <string>
#include "utils/log_config_util.h"

/**
 * 日志管理器：三个行为一致的异步 logger 的初始化 / 获取 / 关闭入口——主日志
 * （LOG_xxx 宏）、get 明细日志（LOG_INFO_GET 宏）与其余操作明细日志
 * （LOG_INFO_OTHER 宏）；pattern / 级别 / 控制台 / 滚动清理统一配置（对三个
 * logger 同款生效），目录与文件名独立配置（见 LogConfigUtil），共用同一异步
 * 线程池。
 */
class LoggerManager {
public:
    static bool Init(const LogConfigUtil &logConfig, std::string &err);
    /** 冲刷并释放全部日志资源；队列残留消息全部落盘后线程池才退出，可重复调用 */
    static void Shutdown();
    /**
     * 进程退出专用收尾：flush 日志并停自有线程池，不调 spdlog::shutdown()——
     * kv client 库线程退出期不终止，优雅关闭会永久阻塞；调用方随后以 std::_Exit
     * 终止进程
     */
    static void ShutdownForExit();
    /** 获取主日志 logger */
    static std::shared_ptr<spdlog::logger> GetLogger() { return logger_; }
    /** 获取 get 明细日志 logger */
    static std::shared_ptr<spdlog::logger> GetGetLogger() { return getLogger_; }
    /** 获取其余操作明细日志 logger（put/exist/delete） */
    static std::shared_ptr<spdlog::logger> GetOtherLogger() { return otherLogger_; }

    LoggerManager() = delete;  // 静态工具类，禁止实例化

private:
    static void CreateLogger(const LogConfigUtil &logConfig, spdlog::level::level_enum level,
                             const char *name, const std::string &dir, const std::string &fileName,
                             std::shared_ptr<spdlog::logger> &out);

    /** 异步日志线程池（单写线程保证各文件内写入顺序） */
    inline static std::shared_ptr<spdlog::details::thread_pool> threadPool_;
    inline static std::shared_ptr<spdlog::logger> logger_;
    inline static std::shared_ptr<spdlog::logger> getLogger_;
    inline static std::shared_ptr<spdlog::logger> otherLogger_;
};

#define LOG_IMPL(LEVEL, ...)                                                                   \
    do {                                                                                       \
        std::shared_ptr<spdlog::logger> logLg_ = ::LoggerManager::GetLogger();                 \
        if (logLg_ != nullptr) {                                                               \
            logLg_->log(spdlog::source_loc{__FILE__, __LINE__, __func__}, LEVEL, __VA_ARGS__); \
        }                                                                                      \
    } while (0)

#define LOG_DEBUG(...) LOG_IMPL(spdlog::level::debug, __VA_ARGS__)
#define LOG_INFO(...) LOG_IMPL(spdlog::level::info, __VA_ARGS__)
#define LOG_WARN(...) LOG_IMPL(spdlog::level::warn, __VA_ARGS__)
#define LOG_ERROR(...) LOG_IMPL(spdlog::level::err, __VA_ARGS__)

#define LOG_INFO_GET(...)                                                                      \
    do {                                                                                       \
        std::shared_ptr<spdlog::logger> getLg_ = ::LoggerManager::GetGetLogger();              \
        if (getLg_ != nullptr) {                                                               \
            getLg_->log(spdlog::source_loc{__FILE__, __LINE__, __func__}, spdlog::level::info, \
                        __VA_ARGS__);                                                          \
        }                                                                                      \
    } while (0)

#define LOG_INFO_OTHER(...)                                                                      \
    do {                                                                                         \
        std::shared_ptr<spdlog::logger> otherLg_ = ::LoggerManager::GetOtherLogger();            \
        if (otherLg_ != nullptr) {                                                               \
            otherLg_->log(spdlog::source_loc{__FILE__, __LINE__, __func__}, spdlog::level::info, \
                          __VA_ARGS__);                                                          \
        }                                                                                        \
    } while (0)
