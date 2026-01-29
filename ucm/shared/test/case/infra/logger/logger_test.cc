
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
#include <ctime>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

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
}  // namespace

class UCLoggerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        CleanDir(test_log_dir_);
        std::filesystem::create_directories(test_log_dir_);
        std::cout << "test_log_path_: " << test_log_path_ << std::endl;
        logger_ = &Logger::GetInstance();
        logger_->Setup(test_log_dir_, 3, 1);  // 3 files, 1MB max size
    }

    static void TearDownTestSuite()
    {
        CleanDir(test_log_dir_);
        spdlog::drop_all();
    }

    static inline std::string test_log_dir_ = "log_test";
    static inline std::string pid_ = std::to_string(getpid());
    static inline std::string test_log_path_ = "log_test/" + pid_ + "/ucm.log";
    static inline Logger* logger_ = nullptr;
};

// Test Make() returns singleton
TEST_F(UCLoggerTest, SingletonBehavior)
{
    Logger& logger1 = Logger::GetInstance();
    Logger& logger2 = Logger::GetInstance();

    ASSERT_EQ(&logger1, &logger2);
}

void VerifyLogFile(const std::string& log_path, const std::string& log_content,
                   bool contains = true)
{
    ASSERT_TRUE(std::filesystem::exists(log_path)) << "Log file does not exist: " << log_path;

    // Verify log content contains the expected message
    std::ifstream log_file(log_path, std::ios::binary);
    ASSERT_TRUE(log_file.is_open()) << "Failed to open log file: " << log_path;

    std::string content((std::istreambuf_iterator<char>(log_file)),
                        std::istreambuf_iterator<char>());
    log_file.close();

    if (contains) {
        ASSERT_TRUE(content.find(log_content) != std::string::npos)
            << "Expected content '" << log_content << "' not found in log file: " << log_path;
    } else {
        ASSERT_TRUE(content.find(log_content) == std::string::npos)
            << "Expected content '" << log_content << "' found in log file: " << log_path;
    }
}

TEST_F(UCLoggerTest, AllLogLevels)
{
    SourceLocation loc{"test_file.cc", "TestFunction", 100};
    std::string debug_msg = "Debug message";
    std::string info_msg = "Info message";
    std::string warn_msg = "Warning message";
    std::string error_msg = "Error message";
    logger_->Log(Level::DEBUG, std::move(loc), std::move(debug_msg));
    logger_->Log(Level::INFO, std::move(loc), std::move(info_msg));
    logger_->Log(Level::WARN, std::move(loc), std::move(warn_msg));
    logger_->Log(Level::ERROR, std::move(loc), std::move(error_msg));
    logger_->Flush();

    VerifyLogFile(test_log_path_, debug_msg, false);  // log level debug was not written to the file
    VerifyLogFile(test_log_path_, info_msg, true);
    VerifyLogFile(test_log_path_, warn_msg, true);
    VerifyLogFile(test_log_path_, error_msg, true);
}

TEST_F(UCLoggerTest, LogCompression)
{
    SourceLocation loc{"test_file.cc", "TestFunction", 100};
    std::string info_msg =
        "Write 25000 log messages to test the number of compressed log files is 3.";
    for (int i = 0; i < 25000; i++) {
        logger_->Log(Level::INFO, std::move(loc), std::move(info_msg));
    }
    logger_->Flush();

    // Count compressed log files (.gz) in test_log_dir_
    int compressed_file_count = 0;
    std::string compressed_file_path = test_log_dir_ + "/" + pid_;
    for (const auto& entry : std::filesystem::directory_iterator(compressed_file_path)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.size() >= 3 && filename.substr(filename.size() - 3) == ".gz") {
                compressed_file_count++;
            }
        }
    }

    ASSERT_EQ(compressed_file_count, 3) << "Expected 3 compressed log files in " << test_log_dir_
                                        << ", but found " << compressed_file_count;
}