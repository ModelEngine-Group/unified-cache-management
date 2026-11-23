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

#include "test/log_test.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include "log/logger_manager.h"
#include "utils/config_util.h"
#include "utils/file_util.h"

namespace {

/** 校验日志文件每行级别均在允许集合内（行格式：[时间] [级别] [线程] 内容），输出总行数 */
bool CheckFileLevels(const std::string &path, const std::vector<std::string> &allowed,
                     size_t &lineCount, std::string &err)
{
    std::vector<std::string> lines;
    if (!FileUtil::ReadLines(path, lines, err)) { return false; }

    lineCount = 0;
    for (const std::string &line : lines) {
        ++lineCount;
        size_t timeBegin = line.find('[');
        size_t levelBegin =
            (timeBegin == std::string::npos) ? std::string::npos : line.find('[', timeBegin + 1);
        size_t levelEnd =
            (levelBegin == std::string::npos) ? std::string::npos : line.find(']', levelBegin);
        if (levelBegin == std::string::npos || levelEnd == std::string::npos) {
            err = "line " + std::to_string(lineCount) + " missing level field";
            return false;
        }

        std::string level = line.substr(levelBegin + 1, levelEnd - levelBegin - 1);
        if (std::find(allowed.begin(), allowed.end(), level) == allowed.end()) {
            err = "line " + std::to_string(lineCount) + " has disallowed level [" + level + "]";
            return false;
        }
    }
    return true;
}

/** 校验单个日志文件并打印结果：可打开、非空（allowEmpty 除外）、行级别合法 */
bool VerifyLogFile(const std::string &path, const std::vector<std::string> &allowed,
                   bool allowEmpty, const std::string &desc)
{
    size_t lineCount = 0;
    std::string err;
    if (!CheckFileLevels(path, allowed, lineCount, err)) {
        std::cout << "[FAIL] " << desc << " (" << path << "): " << err << std::endl;
        return false;
    }
    if (lineCount == 0 && !allowEmpty) {
        std::cout << "[FAIL] " << desc << " (" << path << "): file is empty" << std::endl;
        return false;
    }

    std::cout << "[PASS] " << desc << " (" << path << "): " << lineCount << " lines" << std::endl;
    return true;
}

}  // namespace

bool RunLogTest(const ConfigUtil &config)
{
    // 各级别样例；debug 低于配置级别，应被过滤
    LOG_DEBUG("debug level sample (should be filtered)");
    LOG_INFO("info level sample");
    LOG_WARN("warn level sample: should go to full log and warn log");
    LOG_ERROR("error level sample: should go to full log, warn log and error log");

    // 批量输出触发全量文件按大小滚动，滚动后再补 warn / error 验证专用文件仍正常
    constexpr int kStressCount = 4000;
    for (int i = 0; i < kStressCount; ++i) {
        LOG_INFO("stress log #{:05d}: continuous output to trigger rolling by size", i);
    }
    LOG_WARN("warn level sample after rolling");
    LOG_ERROR("error level sample after rolling");

    LoggerManager::Shutdown();

    bool allPassed = true;
    std::string err;
    std::filesystem::path dirPath(config.GetLogDir());
    std::cout << "---- log files (dir " << dirPath.string() << ") ----" << std::endl;

    // 滚动文件名形如 主名.序号.扩展名（nds_20261112_16.1.log），按展开后的基础名收集全量日志文件
    std::string baseName = config.GetLogFileName();
    std::string baseStem = std::filesystem::path(baseName).stem().string();
    std::string baseExt = std::filesystem::path(baseName).extension().string();
    std::string rollingPrefix = baseStem + ".";
    std::vector<std::string> fullLogFiles;

    std::vector<std::pair<std::string, std::uint64_t>> dirFiles;
    if (!FileUtil::ListFiles(dirPath.string(), dirFiles, err)) {
        std::cout << "[FAIL] " << err << std::endl;
        return false;
    }
    for (const auto &file : dirFiles) {
        const std::string &name = file.first;
        std::cout << "  " << name << "  " << file.second << " bytes" << std::endl;

        bool isFullLog = (name == baseName);
        if (!isFullLog && name.size() > rollingPrefix.size() + baseExt.size() &&
            name.compare(0, rollingPrefix.size(), rollingPrefix) == 0 &&
            name.compare(name.size() - baseExt.size(), baseExt.size(), baseExt) == 0) {
            std::string index = name.substr(rollingPrefix.size(),
                                            name.size() - rollingPrefix.size() - baseExt.size());
            isFullLog = std::all_of(index.begin(), index.end(), [](char c) {
                return std::isdigit(static_cast<unsigned char>(c)) != 0;
            });
        }
        if (isFullLog) { fullLogFiles.push_back((dirPath / name).string()); }
    }

    // 专用文件级别校验（阈值语义）
    allPassed =
        VerifyLogFile((dirPath / config.GetLogWarnFileName()).string(), {"warning", "error"}, false,
                      "warn log contains only warn and above") &&
        allPassed;
    allPassed = VerifyLogFile((dirPath / config.GetLogErrorFileName()).string(), {"error"}, false,
                              "error log contains only error level") &&
                allPassed;

    // 全量文件级别过滤校验（trace / debug 不应出现）
    for (const std::string &file : fullLogFiles) {
        allPassed =
            VerifyLogFile(file, {"info", "warning", "error"}, false, "full log level filter") &&
            allPassed;
    }

    // 文件数量校验：滚动文件上限 + 1 个活跃文件
    size_t fileLimit = static_cast<size_t>(config.GetLogMaxFiles()) + 1;
    if (fullLogFiles.size() > fileLimit) {
        std::cout << "[FAIL] full log file count " << fullLogFiles.size() << " exceeds limit "
                  << fileLimit << std::endl;
        allPassed = false;
    } else {
        std::cout << "[PASS] full log file count " << fullLogFiles.size() << " (limit " << fileLimit
                  << ")" << std::endl;
    }

    std::cout << (allPassed ? "---- log test passed ----" : "---- log test failed ----")
              << std::endl;
    return allPassed;
}
