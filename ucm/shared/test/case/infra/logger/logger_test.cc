
/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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

#include "logger/logger.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <thread>

using namespace UC::Logger;

namespace {
void CleanDir(const std::string& path)
{
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    if (ec) {
        std::cerr << "Failed to remove file: " << path << std::endl;
        std::cerr << "Error: " << ec.message() << std::endl;
        std::exit(1);
    }
}

bool FileContains(const std::filesystem::path& path, const std::string& content)
{
    std::ifstream log_file(path, std::ios::binary);
    if (!log_file.is_open()) { return false; }
    const std::string data((std::istreambuf_iterator<char>(log_file)),
                           std::istreambuf_iterator<char>());
    return data.find(content) != std::string::npos;
}

bool AnyLogFileContains(const std::string& dir, const std::string& prefix,
                        const std::string& content)
{
    if (!std::filesystem::exists(dir)) { return false; }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) { continue; }
        const std::string filename = entry.path().filename().string();
        if (filename.rfind(prefix, 0) == 0 && FileContains(entry.path(), content)) { return true; }
    }
    return false;
}

template <typename Predicate>
bool WaitFor(Predicate&& predicate)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
        if (predicate()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    } while (std::chrono::steady_clock::now() < deadline);
    return predicate();
}
}  // namespace

class UCLoggerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        CleanDir(test_log_dir_);
        std::filesystem::create_directories(test_log_dir_);
        logger_ = &Logger::GetInstance();
        logger_->Setup(test_log_dir_, 3, 1);  // 3 files, 1MB max size
    }

    static void TearDownTestSuite()
    {
        CleanDir(test_log_dir_);
        spdlog::drop_all();
    }

    static inline std::string test_log_dir_ = "log_test";
    static inline Logger* logger_ = nullptr;
};

// Test Make() returns singleton
TEST_F(UCLoggerTest, SingletonBehavior)
{
    Logger& logger1 = Logger::GetInstance();
    Logger& logger2 = Logger::GetInstance();

    ASSERT_EQ(&logger1, &logger2);
}

TEST_F(UCLoggerTest, RegistersLoggerAndLevelFilter)
{
    auto spdlog_logger = spdlog::get("UC");
    ASSERT_NE(spdlog_logger, nullptr);
    EXPECT_FALSE(logger_->IsEnabledFor(Level::DEBUG));
    EXPECT_TRUE(logger_->IsEnabledFor(Level::INFO));
    EXPECT_TRUE(logger_->IsEnabledFor(Level::WARN));
    EXPECT_TRUE(logger_->IsEnabledFor(Level::ERROR));
}

TEST_F(UCLoggerTest, LogEventuallyReachesUcmFile)
{
    const std::string msg = "async ucm logger smoke";
    logger_->Log(Level::WARN, SourceLocation{"logger_test.cc", "LogEventuallyReachesUcmFile", 100},
                 std::string(msg));
    logger_->Flush();

    ASSERT_TRUE(WaitFor([&] { return AnyLogFileContains(test_log_dir_, "ucm-", msg); }));
}

TEST_F(UCLoggerTest, FileOnlyLogEventuallyReachesVllmFile)
{
    const std::string msg = "async vllm logger smoke";
    LogFileOnly(Level::WARN, "logger_test.cc", "FileOnlyLogEventuallyReachesVllmFile", 100,
                std::string(msg));
    Flush();

    ASSERT_TRUE(WaitFor([&] { return AnyLogFileContains(test_log_dir_, "vllm-", msg); }));
}
