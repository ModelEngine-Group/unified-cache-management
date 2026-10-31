/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the Logger class declared in include/kvcache_log.h.
 */
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include "kvcache_log.h"

namespace {

std::string WriteTempLog(const std::string& content)
{
    char tmpl[] = "/tmp/opencode_logger_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) { return {}; }
    if (!content.empty()) {
        ssize_t n = ::write(fd, content.data(), content.size());
        (void)n;
    }
    ::close(fd);
    return tmpl;
}

}  // namespace

TEST(Logger, DisabledByDefaultLoggerIsNoop)
{
    Logger log;  // mEnable=false
    log.log(LogLevel::ERROR, "should not crash\n");
    log.LogWOPrefix(LogLevel::ERROR, "should not crash\n");
    SUCCEED();
}

TEST(Logger, WritesFormattedMessageWhenEnabled)
{
    std::string path = WriteTempLog("");
    ASSERT_FALSE(path.empty());
    {
        Logger log(path, LogLevel::DEBUG, /*enable=*/true);
        log.log(LogLevel::INFO, "hello %d\n", 42);
        log.log(LogLevel::WARNING, "warn %s\n", "x");
        log.log(LogLevel::ERROR, "err\n");
        log.LogWOPrefix(LogLevel::INFO, "raw-chunk\n");
    }
    std::ifstream in(path);
    std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_NE(body.find("hello 42"), std::string::npos);
    EXPECT_NE(body.find("warn x"), std::string::npos);
    EXPECT_NE(body.find("err"), std::string::npos);
    EXPECT_NE(body.find("raw-chunk"), std::string::npos);
    std::remove(path.c_str());
}

TEST(Logger, LevelFilteringSkipsBelowMinimum)
{
    std::string path = WriteTempLog("");
    ASSERT_FALSE(path.empty());
    {
        Logger log(path, LogLevel::WARNING, /*enable=*/true);
        log.log(LogLevel::DEBUG, "debug-should-not-appear\n");
        log.log(LogLevel::INFO, "info-should-not-appear\n");
        log.log(LogLevel::WARNING, "warning-should-appear\n");
        log.log(LogLevel::ERROR, "error-should-appear\n");
    }
    std::ifstream in(path);
    std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(body.find("debug-should-not-appear"), std::string::npos);
    EXPECT_EQ(body.find("info-should-not-appear"), std::string::npos);
    EXPECT_NE(body.find("warning-should-appear"), std::string::npos);
    EXPECT_NE(body.find("error-should-appear"), std::string::npos);
    std::remove(path.c_str());
}

TEST(Logger, SetLevelDynamicallyChangesFilter)
{
    std::string path = WriteTempLog("");
    ASSERT_FALSE(path.empty());
    {
        Logger log(path, LogLevel::ERROR, /*enable=*/true);
        log.log(LogLevel::INFO, "first-info-skip\n");
        log.SetLevel(LogLevel::DEBUG);
        log.log(LogLevel::INFO, "second-info-keep\n");
    }
    std::ifstream in(path);
    std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(body.find("first-info-skip"), std::string::npos);
    EXPECT_NE(body.find("second-info-keep"), std::string::npos);
    std::remove(path.c_str());
}

/* ---- BUG-FOCUS: Logger ctor prints "Fail to open" when the file IS open
 * (inverted condition). See tests/README.md. Documenting current behaviour. ---- */
TEST(LoggerBug, InvertedFailMessageOnSuccess)
{
    std::string path = WriteTempLog("");
    ASSERT_FALSE(path.empty());
    testing::internal::CaptureStderr();
    {
        Logger log(path, LogLevel::INFO, /*enable=*/true);
    }
    std::string err = testing::internal::GetCapturedStderr();
    /* The buggy branch prints the failure banner on success. */
    EXPECT_NE(err.find("Fail to open log file"), std::string::npos);
    std::remove(path.c_str());
}

TEST(Logger, ClosesFileOnDestruction)
{
    std::string path = WriteTempLog("");
    ASSERT_FALSE(path.empty());
    {
        Logger log(path, LogLevel::INFO, /*enable=*/true);
        log.log(LogLevel::INFO, "x\n");
    }
    /* After destruction the file must be reopenable / writable. */
    std::ofstream again(path, std::ios::app);
    EXPECT_TRUE(again.is_open());
    again.close();
    std::remove(path.c_str());
}
