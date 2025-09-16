#ifndef ATB_KV_LOG_H
#define ATB_KV_LOG_H
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <omp.h>
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger 
{
private:
    std::ofstream mLogFile;
    LogLevel mMinLevel;
    std::mutex mMutex;
    bool mEnable;

    static std::string LevelToString(LogLevel level)
    {
        switch (level)
        {
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
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        std::stringstream oss;
        oss << std::put_time(std::localtime(&nowC), "%Y-%m-%d %H:%M:%S");
        oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

public:
    Logger(const std::string &fileName, LogLevel level = LogLevel::INFO, bool enable = true)
        :mMinLevel(level), mEnable(enable)
    {
        if (enable) {
            mLogFile.open(fileName, std::ios::app);
            if (mLogFile.is_open()) {
                std::cerr << "|KVCache Prefetch| Fail to open log file: " << fileName << std::endl;
            }
        }
    }

    Logger(){}

    ~Logger()
    {
        if (mLogFile.is_open()) {
            mLogFile.close();
        }
    }

    void SetLevel(LogLevel level)
    {
        mMinLevel = level;
    }

    void log(LogLevel level, const char* format, ...)
    {
        if (level < mMinLevel || !mLogFile.is_open() || !mEnable) {
            return;
        }
        std::lock_guard<std::mutex> lock(mMutex);
        auto now = std::chrono::system_clock::now();
        auto nowC = std::chrono::system_clock::to_time_t(now);
        auto duration = now.time_since_epoch();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;
        auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count() % 1000;

        std::tm localTime = *std::localtime(&nowC);
        char timeBuffer[26];
        std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", &localTime);
        const char *levelStr = "";
        switch (level)
        {
            case LogLevel::DEBUG: levelStr = "DEBUG"; break;
            case LogLevel::INFO: levelStr = "INFO"; break;
            case LogLevel::WARNING: levelStr = "WARNING"; break;
            case LogLevel::ERROR: levelStr = "ERROR"; break;
            default: levelStr = "UNKNOWN"; break;
        }
        char messageBuffer[4096];
        va_list args;
        va_start(args, format);
        vsnprintf(messageBuffer, sizeof(messageBuffer), format, args);
        va_end(args);

        mLogFile << timeBuffer << "."
                 << std::setfill('0') << std::setw(3) << millis << std::setw(3)
                 << micros << " " << "[" << levelStr << "]"
                 << messageBuffer;
        mLogFile.flush();
    }

    void LogWOPrefix(LogLevel level, const char* format, ...)
    {
        if (level < mMinLevel || !mLogFile.is_open() || !mEnable) {
            return;
        }
        std::lock_guard<std::mutex> lock(mMutex);
        char messageBuffer[2048];
        va_list args;
        va_start(args, format);
        vsnprintf(messageBuffer, sizeof(messageBuffer), format, args);
        va_end(args);
        mLogFile << messageBuffer;
        mLogFile.flush();
    }
};
#endif