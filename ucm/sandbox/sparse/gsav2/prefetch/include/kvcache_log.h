/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
#ifndef ATB_KV_LOG_H
#define ATB_KV_LOG_H
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <sstream>
#include <stdarg.h>
#include <string>
#include <time.h>
enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

class Logger {
public:
    Logger(const std::string& fileName, LogLevel level = LogLevel::INFO, bool enable = true)
        : mMinLevel(level), mEnable(enable)
    {
        if (enable) {
            mLogFile.open(fileName, std::ios::app);
            /* 仅在真实打开失败时打印；此后 log() 因 !is_open() 判别
             * 自动降级为 no-op。 */
            if (!mLogFile.is_open()) {
                std::cerr << "|KVCache Prefetch| Fail to open log file: " << fileName << std::endl;
            }
        }
    }

    Logger() {}

    ~Logger()
    {
        if (mLogFile.is_open()) { mLogFile.close(); }
    }

    void SetLevel(LogLevel level) { mMinLevel = level; }

    /* 调用方在构造昂贵日志实参（如 O(n) 聚合、字符串拼接）前判别，避免
     * 实参求值开销；判别条件与 log() 的早退条件完全一致。 */
    bool enabledFor(LogLevel level) const
    {
        return mEnable && mLogFile.is_open() && level >= mMinLevel;
    }

    void log(LogLevel level, const char* format, ...)
    {
        if (level < mMinLevel || !mLogFile.is_open() || !mEnable) { return; }
        std::lock_guard<std::mutex> lock(mMutex);
        auto now = std::chrono::system_clock::now();
        auto nowC = std::chrono::system_clock::to_time_t(now);
        auto duration = now.time_since_epoch();
        auto millis =
            std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;
        auto micros =
            std::chrono::duration_cast<std::chrono::microseconds>(duration).count() % 1000;

        std::tm localTime;
        localtime_r(&nowC, &localTime);
        char timeBuffer[26];
        std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", &localTime);
        const char* levelStr = "";
        switch (level) {
            case LogLevel::DEBUG: levelStr = "DEBUG"; break;
            case LogLevel::INFO: levelStr = "INFO"; break;
            case LogLevel::WARNING: levelStr = "WARNING"; break;
            case LogLevel::ERROR: levelStr = "ERROR"; break;
            default: levelStr = "UNKNOWN"; break;
        }
        char messageBuffer[4096];
        va_list args;
        va_start(args, format);
        int result = vsnprintf(messageBuffer, sizeof(messageBuffer), format, args);
        va_end(args);
        if (result < 0) {
            mLogFile << "[ERROR] vsnprintf failed\n";
            mLogFile.flush();
            return;
        }
        int three = 3;
        mLogFile << timeBuffer << "." << std::setfill('0') << std::setw(three) << millis
                 << std::setw(three) << micros << " " << "[" << levelStr << "]" << messageBuffer;
        mLogFile.flush();
    }

    void LogWOPrefix(LogLevel level, const char* format, ...)
    {
        if (level < mMinLevel || !mLogFile.is_open() || !mEnable) { return; }
        std::lock_guard<std::mutex> lock(mMutex);
        char messageBuffer[2048];
        va_list args;
        va_start(args, format);
        int result = vsnprintf(messageBuffer, sizeof(messageBuffer), format, args);
        va_end(args);
        if (result < 0) {
            mLogFile << "[ERROR] vsnprintf failed\n";
            mLogFile.flush();
            return;
        }
        mLogFile << messageBuffer;
        mLogFile.flush();
    }

private:
    std::ofstream mLogFile;
    LogLevel mMinLevel;
    std::mutex mMutex;
    bool mEnable;

    static std::string LevelToString(LogLevel level)
    {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }

    static std::string GetTimesTamp()
    {
        auto now = std::chrono::system_clock::now();
        auto nowC = std::chrono::system_clock::to_time_t(now);
        auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        std::stringstream oss;
        int three = 3;
        oss << std::put_time(std::localtime(&nowC), "%Y-%m-%d %H:%M:%S");
        oss << '.' << std::setfill('0') << std::setw(three) << ms.count();
        return oss.str();
    }
};
#endif